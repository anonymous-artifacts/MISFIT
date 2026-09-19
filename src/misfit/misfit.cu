// misfit.cu
//
// MISFIT: the synchronous engine of MISFIT's MIS maintenance algorithm.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <omp.h>
#include <malloc.h>

#include "../../core/FlashGraph.h"
#include "../../core/flash_compactor.h"

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define MAX_BATCH_SIZE 1048576
#define THREADS_PER_BLOCK 256
#define MAX_NEIGHBORS_PER_VERTEX 128
#define MAX_VERTICES_PER_TRANSFER_BATCH 1000000000 // We found this to be optimal number for avoiding hubs

int gpu_device_id;
int SMs;
int mTpSM;

static double g_baseline_used_mb = -1.0;
static double g_peak_delta_mb    = 0.0;
static double g_gpu_total_mb     = 0.0;

static void gpu_memory_set_baseline() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        g_gpu_total_mb     = (double)total_bytes / (1024.0 * 1024.0);
        g_baseline_used_mb = g_gpu_total_mb - (double)free_bytes / (1024.0 * 1024.0);
    }
}

static void gpu_memory_sample() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
    const double total_mb = (double)total_bytes / (1024.0 * 1024.0);
    const double used_mb  = total_mb - (double)free_bytes / (1024.0 * 1024.0);
    double delta_mb = used_mb - g_baseline_used_mb;
    if (delta_mb < 0.0) delta_mb = 0.0;
    if (delta_mb > g_peak_delta_mb) g_peak_delta_mb = delta_mb;
    g_gpu_total_mb = total_mb;
}

struct DetailedTimings {
    float extract_vertices_time;
    float compute_neighborhoods_time;
    float vertex_clustering_time;
    float assign_edges_time;
    float tvb_process_time;
    float clear_assignments_time;
    float cluster_processing_time;
    float total_gpu_kernel_time;
    float host_to_device_time;
    float device_to_host_time;
    float cpu_conversion_time;
    float cpu_graph_update_time;
    float cpu_overhead_time;
    float total_cpu_time;
    float total_gpu_time;
    float total_batch_time;
    float transfer_time;
    int clusters_count;

    DetailedTimings() { reset(); }
    void reset() {
        extract_vertices_time = compute_neighborhoods_time = vertex_clustering_time = assign_edges_time = 0;
        tvb_process_time = clear_assignments_time = cluster_processing_time = total_gpu_kernel_time = 0;
        host_to_device_time = device_to_host_time = cpu_conversion_time = cpu_graph_update_time = cpu_overhead_time = 0;
        total_cpu_time = total_gpu_time = total_batch_time = transfer_time = clusters_count = 0;
    }
    void calculateAggregates() {
        cluster_processing_time = extract_vertices_time + compute_neighborhoods_time + vertex_clustering_time + assign_edges_time + clear_assignments_time;

        total_gpu_kernel_time = cluster_processing_time + tvb_process_time;

        transfer_time = host_to_device_time + device_to_host_time;

        total_cpu_time = transfer_time + cpu_conversion_time + cpu_graph_update_time + cpu_overhead_time;

        total_gpu_time = total_gpu_kernel_time + transfer_time;
    }
    void accumulate(const DetailedTimings& other) {
        extract_vertices_time += other.extract_vertices_time;
        compute_neighborhoods_time += other.compute_neighborhoods_time;
        vertex_clustering_time += other.vertex_clustering_time;
        assign_edges_time += other.assign_edges_time;
        tvb_process_time += other.tvb_process_time;
        clear_assignments_time += other.clear_assignments_time;
        cluster_processing_time += other.cluster_processing_time;
        total_gpu_kernel_time += other.total_gpu_kernel_time;
        host_to_device_time += other.host_to_device_time;
        device_to_host_time += other.device_to_host_time;
        cpu_conversion_time += other.cpu_conversion_time;
        cpu_graph_update_time += other.cpu_graph_update_time;
        cpu_overhead_time += other.cpu_overhead_time;
        total_cpu_time += other.total_cpu_time;
        total_gpu_time += other.total_gpu_time;
        total_batch_time += other.total_batch_time;
        transfer_time += other.transfer_time;
        clusters_count += other.clusters_count;
    }
};

struct __align__(8) DeviceNode {
    bool membership;
    int cluster_id;
    __device__ __host__ DeviceNode() : membership(false), cluster_id(-1) {}
    __device__ __host__ DeviceNode(bool m, int c) : membership(m), cluster_id(c) {}
};

struct __align__(16) DeviceEdge {
    int source;
    int destination;
    bool isInsertion;
    int clusterID;
    __device__ __host__ DeviceEdge() : source(0), destination(0), isInsertion(true), clusterID(-1) {}
    __device__ __host__ DeviceEdge(int s, int d, bool ins) : source(s), destination(d), isInsertion(ins), clusterID(-1) {}
};

struct DeviceVertexCluster {
    int edge_head;
    int edge_count;
    __device__ __host__ DeviceVertexCluster() : edge_head(-1), edge_count(0) {}
};

struct Edge {
    int source, destination;
    bool isInsertion;
    int clusterID;
    Edge(int s, int d, bool ins) : source(s), destination(d), isInsertion(ins), clusterID(-1) {}
};

// Exact 2-hop neighborhood of a single seed vertex
vector<int> get_h_hop_neighborhood(int start_vertex, HybridGraph* graph, int hop_limit,
                                    vector<char>& visited, vector<int>& tracker,
                                    vector<int>& current_frontier, vector<int>& next_frontier,
                                    vector<int>& neighbor_scratch,
                                    unordered_map<int, vector<int>>* hop1_rows) {
    vector<int> neighborhood;
    tracker.clear();
    current_frontier.clear();
    next_frontier.clear();

    if (start_vertex >= 0 && start_vertex < graph->num_nodes) {
        visited[start_vertex] = 1;
        tracker.push_back(start_vertex);
        current_frontier.push_back(start_vertex);
        neighborhood.push_back(start_vertex);
    }

    for (int hop = 0; hop < hop_limit && hop < 2; ++hop) {
        if (current_frontier.empty()) break;
        for (int curr : current_frontier) {
            graph->neighbors_into(curr, neighbor_scratch);
            if (hop == 1 && hop1_rows != nullptr && hop1_rows->find(curr) == hop1_rows->end()) {
                (*hop1_rows)[curr] = neighbor_scratch;
            }
            for (int neighbor : neighbor_scratch) {
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    tracker.push_back(neighbor);
                    next_frontier.push_back(neighbor);
                    neighborhood.push_back(neighbor);
                }
            }
        }
        current_frontier = next_frontier;
        next_frontier.clear();
    }

    // Clean up visited array using the tracker for O(K) complexity
    for (int v : tracker) {
        visited[v] = 0;
    }

    return neighborhood;
}

__global__ void init_nodes_kernel(DeviceNode* nodes, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        nodes[idx] = DeviceNode(false, -1);
    }
}

__global__ void build_vertex_map_kernel(const int* mapped_vertices, int num_vertices, int* vertex_to_idx_map) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        int v = mapped_vertices[idx];
        vertex_to_idx_map[v] = idx;
    }
}

__global__ void vertex_clustering_csr_kernel(const int* mapped_vertices, int num_vertices,const long long* neighborhood_offsets, const int* neighborhood_elements,DeviceNode* nodes, DeviceVertexCluster* clusters, int* cluster_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    int vertex = mapped_vertices[tid];
    if (nodes[vertex].cluster_id != -1) return;

    long long start = neighborhood_offsets[tid];
    long long end = neighborhood_offsets[tid + 1];

    int overlapping_clusters[4];
    int overlap_count = 0;

    for (long long i = start; i < end && overlap_count < 4; i++) {
        int neighbor = neighborhood_elements[i];
        int c_id = nodes[neighbor].cluster_id;
        if (c_id != -1) {
            bool found = false;
            for (int j = 0; j < overlap_count; j++) {
                if (overlapping_clusters[j] == c_id) { found = true; break; }
            }
            if (!found) overlapping_clusters[overlap_count++] = c_id;
        }
    }

    int final_cluster_id;
    if (overlap_count == 0) {
        final_cluster_id = atomicAdd(cluster_count, 1);
        if (final_cluster_id < MAX_BATCH_SIZE) {
            clusters[final_cluster_id].edge_head = -1;
            clusters[final_cluster_id].edge_count = 0;
        }
    } else {
        final_cluster_id = overlapping_clusters[0];
    }

    for (long long i = start; i < end; i++) {
        int neighbor = neighborhood_elements[i];
        nodes[neighbor].cluster_id = final_cluster_id;
    }
}

__global__ void assign_edges_to_clusters_kernel(const DeviceEdge* edges, int num_edges,DeviceNode* nodes, DeviceVertexCluster* clusters, int* edge_next) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    const DeviceEdge& edge = edges[tid];
    int cluster_id = -1;
    if (nodes[edge.source].cluster_id != -1) cluster_id = nodes[edge.source].cluster_id;
    else if (nodes[edge.destination].cluster_id != -1) cluster_id = nodes[edge.destination].cluster_id;

    if (cluster_id != -1 && cluster_id < MAX_BATCH_SIZE) {
        int old_head = atomicExch(&clusters[cluster_id].edge_head, tid);
        edge_next[tid] = old_head;
        atomicAdd(&clusters[cluster_id].edge_count, 1);
    }
}

__global__ void process_clusters_csr_kernel(const DeviceEdge* edges, DeviceVertexCluster* clusters, const int* cluster_count_ptr, const int* edge_next, DeviceNode* nodes, const long long* neighborhood_offsets, const int* neighborhood_elements,const int* vertex_to_idx_map, int total_nodes) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= *cluster_count_ptr) return;

    int curr_edge = clusters[cluster_id].edge_head;
    while (curr_edge != -1) {
        const DeviceEdge& edge = edges[curr_edge];
        int u = edge.source;
        int v = edge.destination;

        bool u_in_mis = nodes[u].membership;
        bool v_in_mis = nodes[v].membership;

        if (edge.isInsertion) {
            if (u_in_mis && v_in_mis) {
                int to_remove = (u > v) ? u : v;
                nodes[to_remove].membership = false;

                int idx = vertex_to_idx_map[to_remove];
                if(idx != -1) {
                    long long start = neighborhood_offsets[idx];
                    long long end = neighborhood_offsets[idx + 1];
                    int count = (int)(end - start);
                    for (int j = 0; j < count; j++) {
                        int neighbor = neighborhood_elements[start + j];
                        if (!nodes[neighbor].membership) {
                            int nbr_idx = vertex_to_idx_map[neighbor];
                            if (nbr_idx == -1) continue;
                            bool can_add = true;
                            long long n_start = neighborhood_offsets[nbr_idx];
                            long long n_end = neighborhood_offsets[nbr_idx + 1];
                            int count_k = (int)(n_end - n_start);
                            for (int k = 0; k < count_k; k++) {
                                int n_nbr = neighborhood_elements[n_start + k];
                                if (nodes[n_nbr].membership) { can_add = false; break; }
                            }
                            if (can_add) { nodes[neighbor].membership = true; break; }
                        }
                    }
                }
            }
        } else {
            if (u_in_mis && !v_in_mis) {
                bool can_add = true;
                int idx = vertex_to_idx_map[v];
                if(idx != -1) {
                    long long start = neighborhood_offsets[idx];
                    long long end = neighborhood_offsets[idx + 1];
                    int count = (int)(end - start);
                    for (int j = 0; j < count; j++) {
                        int neighbor = neighborhood_elements[start + j];
                        if (neighbor != u && nodes[neighbor].membership) { can_add = false; break; }
                    }
                }
                if (can_add) nodes[v].membership = true;
            } else if (!u_in_mis && v_in_mis) {
                bool can_add = true;
                int idx = vertex_to_idx_map[u];
                if(idx != -1) {
                    long long start = neighborhood_offsets[idx];
                    long long end = neighborhood_offsets[idx + 1];
                    int count = (int)(end - start);
                    for (int j = 0; j < count; j++) {
                        int neighbor = neighborhood_elements[start + j];
                        if (neighbor != v && nodes[neighbor].membership) { can_add = false; break; }
                    }
                }
                if (can_add) nodes[u].membership = true;
            }
        }
        curr_edge = edge_next[curr_edge];
    }
}

__global__ void clear_cluster_assignments_kernel(const int* mapped_vertices, int vertex_count, DeviceNode* nodes, int* vertex_to_idx_map) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vertex_count) {
        int vertex = mapped_vertices[tid];
        nodes[vertex].cluster_id = -1;
        vertex_to_idx_map[vertex] = -1;
    }
}

class GPUVertexClusteringGraph {
    private:
        DeviceNode* d_nodes;
        int total_nodes;

        static const int NUM_CHANNELS = 2;
        long long MAX_CSR_ELEMENTS;
        int current_channel;

        DeviceEdge* d_edges_pool[NUM_CHANNELS];
        int* d_mapped_vertices_pool[NUM_CHANNELS];
        long long* d_neighborhood_offsets_pool[NUM_CHANNELS];
        int* d_neighborhood_elements_pool[NUM_CHANNELS];
        int* d_cluster_count[NUM_CHANNELS];
        DeviceVertexCluster* d_clusters_pool[NUM_CHANNELS];
        int* d_edge_next[NUM_CHANNELS];
        cudaEvent_t channel_sync_event[NUM_CHANNELS];

        DeviceEdge* h_edges_pool[NUM_CHANNELS];
        int* h_mapped_vertices_pool[NUM_CHANNELS];
        long long* h_neighborhood_offsets_pool[NUM_CHANNELS];
        int* h_neighborhood_elements_pool[NUM_CHANNELS];

        int* d_vertex_to_idx_map;
        cudaStream_t stream;

        int* h_subbatch_counts;

    public:
        GPUVertexClusteringGraph(int total_nodes_in, long long max_csr_elements_in) {
            total_nodes = total_nodes_in;
            CUDA_CHECK(cudaMalloc(&d_nodes, (size_t)total_nodes * sizeof(DeviceNode)));

            MAX_CSR_ELEMENTS = max_csr_elements_in;
            current_channel = 0;

            CUDA_CHECK(cudaMalloc(&d_vertex_to_idx_map, (size_t)total_nodes * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_vertex_to_idx_map, -1, (size_t)total_nodes * sizeof(int)));
            CUDA_CHECK(cudaStreamCreate(&stream));

            CUDA_CHECK(cudaMallocHost(&h_subbatch_counts, 1000 * sizeof(int)));

            for(int i = 0; i < NUM_CHANNELS; i++) {
                CUDA_CHECK(cudaMalloc(&d_edges_pool[i], MAX_BATCH_SIZE * sizeof(DeviceEdge)));
                CUDA_CHECK(cudaMalloc(&d_mapped_vertices_pool[i], MAX_BATCH_SIZE * 2 * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_neighborhood_offsets_pool[i], (MAX_BATCH_SIZE * 2 + 1) * sizeof(long long)));
                CUDA_CHECK(cudaMalloc(&d_neighborhood_elements_pool[i], MAX_CSR_ELEMENTS * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_cluster_count[i], sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_clusters_pool[i], MAX_BATCH_SIZE * sizeof(DeviceVertexCluster)));
                CUDA_CHECK(cudaMalloc(&d_edge_next[i], MAX_BATCH_SIZE * sizeof(int)));

                CUDA_CHECK(cudaMallocHost(&h_edges_pool[i], MAX_BATCH_SIZE * sizeof(DeviceEdge)));
                CUDA_CHECK(cudaMallocHost(&h_mapped_vertices_pool[i], MAX_BATCH_SIZE * 2 * sizeof(int)));
                CUDA_CHECK(cudaMallocHost(&h_neighborhood_offsets_pool[i], (MAX_BATCH_SIZE * 2 + 1) * sizeof(long long)));
                CUDA_CHECK(cudaMallocHost(&h_neighborhood_elements_pool[i], MAX_CSR_ELEMENTS * sizeof(int)));

                CUDA_CHECK(cudaEventCreate(&channel_sync_event[i]));
                CUDA_CHECK(cudaEventRecord(channel_sync_event[i], stream));
            }

            initializeNodes();
        }

        ~GPUVertexClusteringGraph() {
            cudaStreamSynchronize(stream);
            cudaFree(d_nodes);
            cudaFree(d_vertex_to_idx_map);
            for(int i = 0; i < NUM_CHANNELS; i++) {
                cudaFree(d_edges_pool[i]); cudaFree(d_mapped_vertices_pool[i]);
                cudaFree(d_neighborhood_offsets_pool[i]); cudaFree(d_neighborhood_elements_pool[i]);
                cudaFree(d_cluster_count[i]); cudaFree(d_clusters_pool[i]); cudaFree(d_edge_next[i]);

                cudaFreeHost(h_edges_pool[i]); cudaFreeHost(h_mapped_vertices_pool[i]);
                cudaFreeHost(h_neighborhood_offsets_pool[i]); cudaFreeHost(h_neighborhood_elements_pool[i]);

                cudaEventDestroy(channel_sync_event[i]);
            }
            cudaFreeHost(h_subbatch_counts);
            cudaStreamDestroy(stream);
        }

        void initializeNodes() {
            int threads = THREADS_PER_BLOCK;
            int blocks = (total_nodes + threads - 1) / threads;
            init_nodes_kernel<<<blocks, threads, 0, stream>>>(d_nodes, total_nodes);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void setMISFromHost(const vector<bool>& mis) {
            vector<DeviceNode> host_nodes(total_nodes);
            for (int i = 0; i < total_nodes; i++) host_nodes[i] = DeviceNode(mis[i], -1);
            CUDA_CHECK(cudaMemcpyAsync(d_nodes, host_nodes.data(), total_nodes * sizeof(DeviceNode), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        vector<bool> getMISToHost() {
            vector<DeviceNode> host_nodes(total_nodes);
            CUDA_CHECK(cudaMemcpyAsync(host_nodes.data(), d_nodes, total_nodes * sizeof(DeviceNode), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            vector<bool> mis(total_nodes);
            for (int i = 0; i < total_nodes; i++) mis[i] = host_nodes[i].membership;
            return mis;
        }

        void synchronizeChannels() {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void processBatchWithVertexClustering(const vector<DeviceEdge>& batch, int hop_limit, HybridGraph* cpu_graph, DetailedTimings& timings) {
            if (batch.empty()) return;
            timings.reset();

            struct SubBatchEvents {
                cudaEvent_t start_transfer, stop_transfer;
                cudaEvent_t start_map, stop_map;
                cudaEvent_t start_cluster, stop_cluster;
                cudaEvent_t start_assign, stop_assign;
                cudaEvent_t start_tvb, stop_tvb;
                cudaEvent_t start_clear, stop_clear;
            };
            vector<SubBatchEvents> events_list;

            int current_edge_idx = 0;
            int total_edges = batch.size();
            int sub_batch_index = 0;

            while (current_edge_idx < total_edges) {
                // Wait for the channel to be free before CPU writes to it
                CUDA_CHECK(cudaEventSynchronize(channel_sync_event[current_channel]));

                int sub_batch_edges = 0;
                int sub_batch_vertices = 0;
                long long current_csr_elements = 0;

                auto extract_start = chrono::high_resolution_clock::now();
                unordered_set<int> unique_vertices;
                h_neighborhood_offsets_pool[current_channel][0] = 0;

                while (current_edge_idx < total_edges && sub_batch_edges < MAX_BATCH_SIZE) {
                    const DeviceEdge& edge = batch[current_edge_idx];

                    long long estimated_new_elements = 0;
                    if (unique_vertices.find(edge.source) == unique_vertices.end()) {
                        estimated_new_elements += ((long long)cpu_graph->degree(edge.source) * (hop_limit>1 ? 1000 : 1));
                    }
                    if (unique_vertices.find(edge.destination) == unique_vertices.end()) {
                        estimated_new_elements += ((long long)cpu_graph->degree(edge.destination) * (hop_limit>1 ? 1000 : 1));
                    }

                    if (sub_batch_edges > 0) {
                        if (current_csr_elements + estimated_new_elements >= MAX_CSR_ELEMENTS) break;
                    }

                    current_csr_elements += estimated_new_elements;

                    unique_vertices.insert(edge.source);
                    unique_vertices.insert(edge.destination);
                    h_edges_pool[current_channel][sub_batch_edges] = edge;

                    sub_batch_edges++;
                    current_edge_idx++;
                }

                vector<int> unique_v_vec(unique_vertices.begin(), unique_vertices.end());
                auto extract_end = chrono::high_resolution_clock::now();
                timings.extract_vertices_time += chrono::duration_cast<chrono::microseconds>(extract_end - extract_start).count() / 1000.0;

                auto cpu_start = chrono::high_resolution_clock::now();

                vector<vector<int>> exact_neighborhoods(unique_v_vec.size());
                vector<unordered_map<int, vector<int>>> per_thread_hop1_rows(omp_get_max_threads());

                #pragma omp parallel
                {
                    vector<char> local_visited(cpu_graph->num_nodes, 0);
                    vector<int> local_tracker;
                    vector<int> local_current;
                    vector<int> local_next;
                    vector<int> local_neighbor_scratch;
                    int tid = omp_get_thread_num();

                    #pragma omp for schedule(dynamic)
                    for (size_t i = 0; i < unique_v_vec.size(); i++) {
                        exact_neighborhoods[i] = get_h_hop_neighborhood(
                            unique_v_vec[i], cpu_graph, hop_limit,
                            local_visited, local_tracker, local_current, local_next, local_neighbor_scratch,
                            &per_thread_hop1_rows[tid]);
                    }
                }

                unordered_map<int, vector<int>> candidate_rows;
                for (auto& tmap : per_thread_hop1_rows) {
                    for (auto& kv : tmap) {
                        if (unique_vertices.count(kv.first)) continue;
                        if (!candidate_rows.count(kv.first)) {
                            candidate_rows.emplace(kv.first, std::move(kv.second));
                        }
                    }
                }

                const size_t max_total_rows = (size_t)MAX_BATCH_SIZE * 2;

                auto pack_row = [&](int v, const vector<int>& row) {
                    h_mapped_vertices_pool[current_channel][sub_batch_vertices] = v;
                    long long offset_start = h_neighborhood_offsets_pool[current_channel][sub_batch_vertices];
                    if (offset_start + (long long)row.size() >= MAX_CSR_ELEMENTS) {
                        long long old_max = MAX_CSR_ELEMENTS;
                        MAX_CSR_ELEMENTS = (offset_start + row.size()) * 1.5;
                        cout << "  [Warning] Resizing MAX_CSR_ELEMENTS from " << old_max << " to " << MAX_CSR_ELEMENTS << " elements." << endl;
                        CUDA_CHECK(cudaDeviceSynchronize());
                        for(int k = 0; k < NUM_CHANNELS; k++) {
                            int* new_h;
                            int* new_d;
                            CUDA_CHECK(cudaMallocHost(&new_h, MAX_CSR_ELEMENTS * sizeof(int)));
                            CUDA_CHECK(cudaMalloc(&new_d, MAX_CSR_ELEMENTS * sizeof(int)));
                            if (k == current_channel && offset_start > 0) {
                                memcpy(new_h, h_neighborhood_elements_pool[k], offset_start * sizeof(int));
                            }
                            CUDA_CHECK(cudaFreeHost(h_neighborhood_elements_pool[k]));
                            CUDA_CHECK(cudaFree(d_neighborhood_elements_pool[k]));
                            h_neighborhood_elements_pool[k] = new_h;
                            d_neighborhood_elements_pool[k] = new_d;
                        }
                    }
                    for(long long j = 0; j < (long long)row.size(); j++) {
                        h_neighborhood_elements_pool[current_channel][offset_start + j] = row[j];
                    }
                    current_csr_elements += row.size();
                    h_neighborhood_offsets_pool[current_channel][sub_batch_vertices + 1] = current_csr_elements;
                    sub_batch_vertices++;
                };

                current_csr_elements = 0; // Reset for packing
                for (size_t i = 0; i < unique_v_vec.size(); i++) {
                    pack_row(unique_v_vec[i], exact_neighborhoods[i]);
                }
                for (const auto& kv : candidate_rows) {
                    if ((size_t)sub_batch_vertices >= max_total_rows) break;
                    pack_row(kv.first, kv.second);
                }

                auto cpu_end = chrono::high_resolution_clock::now();
                timings.cpu_overhead_time += chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

                SubBatchEvents ev;
                CUDA_CHECK(cudaEventCreate(&ev.start_transfer)); CUDA_CHECK(cudaEventCreate(&ev.stop_transfer));
                CUDA_CHECK(cudaEventCreate(&ev.start_map)); CUDA_CHECK(cudaEventCreate(&ev.stop_map));
                CUDA_CHECK(cudaEventCreate(&ev.start_cluster)); CUDA_CHECK(cudaEventCreate(&ev.stop_cluster));
                CUDA_CHECK(cudaEventCreate(&ev.start_assign)); CUDA_CHECK(cudaEventCreate(&ev.stop_assign));
                CUDA_CHECK(cudaEventCreate(&ev.start_tvb)); CUDA_CHECK(cudaEventCreate(&ev.stop_tvb));
                CUDA_CHECK(cudaEventCreate(&ev.start_clear)); CUDA_CHECK(cudaEventCreate(&ev.stop_clear));

                CUDA_CHECK(cudaEventRecord(ev.start_transfer, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_edges_pool[current_channel], h_edges_pool[current_channel], sub_batch_edges * sizeof(DeviceEdge), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_mapped_vertices_pool[current_channel], h_mapped_vertices_pool[current_channel], sub_batch_vertices * sizeof(int), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_neighborhood_offsets_pool[current_channel], h_neighborhood_offsets_pool[current_channel], (sub_batch_vertices + 1) * sizeof(long long), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_neighborhood_elements_pool[current_channel], h_neighborhood_elements_pool[current_channel], current_csr_elements * sizeof(int), cudaMemcpyHostToDevice, stream));

                int zero = 0;
                CUDA_CHECK(cudaMemcpyAsync(d_cluster_count[current_channel], &zero, sizeof(int), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaEventRecord(ev.stop_transfer, stream));

                int threads = THREADS_PER_BLOCK;
                int blocks_v = (sub_batch_vertices + threads - 1) / threads;
                int blocks_e = (sub_batch_edges + threads - 1) / threads;
                int blocks_c = (MAX_BATCH_SIZE + threads - 1) / threads;

                // Build Map
                CUDA_CHECK(cudaEventRecord(ev.start_map, stream));
                build_vertex_map_kernel<<<blocks_v, threads, 0, stream>>>(d_mapped_vertices_pool[current_channel], sub_batch_vertices, d_vertex_to_idx_map);
                CUDA_CHECK(cudaEventRecord(ev.stop_map, stream));

                CUDA_CHECK(cudaEventRecord(ev.start_cluster, stream));
                vertex_clustering_csr_kernel<<<blocks_v, threads, 0, stream>>>(
                    d_mapped_vertices_pool[current_channel], sub_batch_vertices, d_neighborhood_offsets_pool[current_channel], d_neighborhood_elements_pool[current_channel],
                    d_nodes, d_clusters_pool[current_channel], d_cluster_count[current_channel]);
                CUDA_CHECK(cudaEventRecord(ev.stop_cluster, stream));

                CUDA_CHECK(cudaEventRecord(ev.start_assign, stream));
                assign_edges_to_clusters_kernel<<<blocks_e, threads, 0, stream>>>(
                    d_edges_pool[current_channel], sub_batch_edges, d_nodes, d_clusters_pool[current_channel], d_edge_next[current_channel]);
                CUDA_CHECK(cudaEventRecord(ev.stop_assign, stream));

                CUDA_CHECK(cudaEventRecord(ev.start_tvb, stream));
                process_clusters_csr_kernel<<<blocks_c, threads, 0, stream>>>(
                    d_edges_pool[current_channel], d_clusters_pool[current_channel], d_cluster_count[current_channel], d_edge_next[current_channel], d_nodes,
                    d_neighborhood_offsets_pool[current_channel], d_neighborhood_elements_pool[current_channel], d_vertex_to_idx_map, total_nodes);
                CUDA_CHECK(cudaEventRecord(ev.stop_tvb, stream));

                // Fetch the count asynchronously into the pinned array
                if (sub_batch_index < 1000) {
                    CUDA_CHECK(cudaMemcpyAsync(&h_subbatch_counts[sub_batch_index], d_cluster_count[current_channel], sizeof(int), cudaMemcpyDeviceToHost, stream));
                }

                CUDA_CHECK(cudaEventRecord(ev.start_clear, stream));
                clear_cluster_assignments_kernel<<<blocks_v, threads, 0, stream>>>(
                    d_mapped_vertices_pool[current_channel], sub_batch_vertices, d_nodes, d_vertex_to_idx_map);
                CUDA_CHECK(cudaEventRecord(ev.stop_clear, stream));

                // Record the sync event for this channel
                CUDA_CHECK(cudaEventRecord(channel_sync_event[current_channel], stream));

                events_list.push_back(ev);
                current_channel = (current_channel + 1) % NUM_CHANNELS;
                sub_batch_index++;
            }

            // Sync stream to gather batch results asynchronously
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int ev_idx = 0;
            for (const auto& ev : events_list) {
                float elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev.start_transfer, ev.stop_transfer)); timings.host_to_device_time += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev.start_map, ev.stop_map)); timings.compute_neighborhoods_time += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev.start_cluster, ev.stop_cluster)); timings.vertex_clustering_time += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev.start_assign, ev.stop_assign)); timings.assign_edges_time += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev.start_tvb, ev.stop_tvb)); timings.tvb_process_time += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev.start_clear, ev.stop_clear)); timings.clear_assignments_time += elapsed;

                if (ev_idx < 1000) {
                    timings.clusters_count += h_subbatch_counts[ev_idx];
                }

                CUDA_CHECK(cudaEventDestroy(ev.start_transfer)); CUDA_CHECK(cudaEventDestroy(ev.stop_transfer));
                CUDA_CHECK(cudaEventDestroy(ev.start_map)); CUDA_CHECK(cudaEventDestroy(ev.stop_map));
                CUDA_CHECK(cudaEventDestroy(ev.start_cluster)); CUDA_CHECK(cudaEventDestroy(ev.stop_cluster));
                CUDA_CHECK(cudaEventDestroy(ev.start_assign)); CUDA_CHECK(cudaEventDestroy(ev.stop_assign));
                CUDA_CHECK(cudaEventDestroy(ev.start_tvb)); CUDA_CHECK(cudaEventDestroy(ev.stop_tvb));
                CUDA_CHECK(cudaEventDestroy(ev.start_clear)); CUDA_CHECK(cudaEventDestroy(ev.stop_clear));
                ev_idx++;
            }
        }
};

vector<DeviceEdge> convertToDeviceEdges(const vector<Edge>& host_edges) {
    vector<DeviceEdge> device_edges;
    device_edges.reserve(host_edges.size());
    for (const auto& edge : host_edges) {
        device_edges.emplace_back(edge.source, edge.destination, edge.isInsertion);
    }
    return device_edges;
}

void processBatchGPUVertexClustering(vector<Edge>& batch, HybridGraph& graph, GPUVertexClusteringGraph& gpu_graph, int hop_limit, DetailedTimings& batch_timings) {
    auto total_start = chrono::high_resolution_clock::now();

    auto graph_update_start = chrono::high_resolution_clock::now();
    for (const auto& edge : batch) {
        if (edge.isInsertion && !graph.isAdjacent(edge.source, edge.destination)) {
            graph.insertEdge(edge.source, edge.destination);
        } else if (!edge.isInsertion && graph.isAdjacent(edge.source, edge.destination)) {
            graph.removeEdge(edge.source, edge.destination);
        }
    }
    auto graph_update_end = chrono::high_resolution_clock::now();
    batch_timings.cpu_graph_update_time = chrono::duration_cast<chrono::microseconds>(graph_update_end - graph_update_start).count() / 1000.0;

    auto conversion_start = chrono::high_resolution_clock::now();
    vector<DeviceEdge> device_batch = convertToDeviceEdges(batch);
    auto conversion_end = chrono::high_resolution_clock::now();
    batch_timings.cpu_conversion_time = chrono::duration_cast<chrono::microseconds>(conversion_end - conversion_start).count() / 1000.0;

    // Asynchronously queue the batch sub-batches
    gpu_graph.processBatchWithVertexClustering(device_batch, hop_limit, &graph, batch_timings);

    auto total_end = chrono::high_resolution_clock::now();
    batch_timings.total_batch_time = chrono::duration_cast<chrono::microseconds>(total_end - total_start).count() / 1000.0;
    batch_timings.calculateAggregates();

}

int main(int argc, char* argv[]) {
    #ifdef M_MMAP_MAX
    mallopt(M_MMAP_MAX, 0); // Prevent glibc from exhausting OS virtual memory maps with large degree vectors
    #endif

    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " <graph.egr> <initial_mis.txt> <batch_folder> <num_batches> <gpu_device_id>" << endl;
        cerr << "Note: <graph.egr> must be a pre-built binary CSR file (see ECLgraph.h format)." << endl;
        cerr << "      If you only have a .mtx file, first run:" << endl;
        cerr << "        python3 tools/efficient_converter.py <graph.mtx> <graph.egr>" << endl;
        return 1;
    }
    cout << "=========================================START=====================================================" << endl;
    cout << "======================================= MISFIT (SYNC) =========================================" << endl;

    string graph_path(argv[1]);
    if (graph_path.size() < 4 || graph_path.substr(graph_path.size() - 4) != ".egr") {
        cerr << "[Warning] " << graph_path << " does not end in .egr — this binary expects a pre-built binary CSR file." << endl;
        cerr << "          If this fails to open, run: python3 tools/efficient_converter.py <graph.mtx> <graph.egr>" << endl;
    }

    gpu_device_id = stoi(argv[5]);
    CUDA_CHECK(cudaSetDevice(gpu_device_id));
    CUDA_CHECK(cudaFree(0)); // Initialize the CUDA context immediately while OS RAM is completely free
    gpu_memory_set_baseline(); // capture baseline before any of our own allocations

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpu_device_id));
    SMs = deviceProp.multiProcessorCount;
    mTpSM = deviceProp.maxThreadsPerMultiProcessor;

    // Opening a FlashGraph is O(1) regardless of file size.
    HybridGraph graph;
    graph.open(argv[1]);
    int num_nodes = graph.num_nodes;

    double avg_degree = num_nodes > 0 ? (double)graph.base.edges / num_nodes : 1.0;
    long long estimated_csr_elements = (long long)(MAX_BATCH_SIZE * 2.0 * avg_degree * avg_degree);
    const long long DEFAULT_MAX_CSR_ELEMENTS = 50000000LL; // conservative fallback (~200MB/array)
    long long max_csr_elements = estimated_csr_elements > 0 ? min(800000000LL, estimated_csr_elements) : DEFAULT_MAX_CSR_ELEMENTS;
    max_csr_elements = max(max_csr_elements, 1000000LL); // sane floor; dynamic growth handles underestimates
    cout << "Graph average degree: " << avg_degree << ", MAX_CSR_ELEMENTS set to: " << max_csr_elements << endl;

    GPUVertexClusteringGraph gpu_graph(num_nodes, max_csr_elements);
    gpu_memory_sample(); // captures the double-buffered pinned/device pools just allocated

    ifstream inputFile(argv[2]);
    int initial_card = 0;
    vector<bool> initial_mis(num_nodes, false);
    if (inputFile.is_open()) {
        string line;
        while (getline(inputFile, line)) {
            int node_id = stoi(line) - 1;
            if (node_id >= 0 && node_id < num_nodes) {
                initial_mis[node_id] = true;
                initial_card++;
            }
        }
        inputFile.close();
    }

    gpu_graph.setMISFromHost(initial_mis);
    cout << "Initial MIS cardinality: " << initial_card << endl;

    string folderPath(argv[3]);
    
    int requested_num_batches = stoi(argv[4]);
    vector<vector<Edge>> batches_of_edges;
    vector<Edge> batch_of_edges;

    vector<filesystem::path> batch_paths;
    for (const auto& entry : filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            batch_paths.push_back(entry.path());
        }
    }
    sort(batch_paths.begin(), batch_paths.end(), [](const filesystem::path& a, const filesystem::path& b) {
        return stoi(a.stem().string()) < stoi(b.stem().string());
    });

    for (const auto& path : batch_paths) {
        if ((int)batches_of_edges.size() >= requested_num_batches) break;
        ifstream file(path);
        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            int src, dest;
            if (iss >> src >> dest) {
                src--; dest--;
                if (src >= 0 && src < graph.num_nodes && dest >= 0 && dest < graph.num_nodes && src != dest) {
                    batch_of_edges.emplace_back(src, dest, !graph.isAdjacent(src, dest));
                }
            }
        }
        batches_of_edges.push_back(move(batch_of_edges));
        batch_of_edges.clear();
    }

    int num_insertion_batches = min(requested_num_batches, (int)batches_of_edges.size());
    int hop_limit = 2;
    double time_cum = 0;

    DetailedTimings cumulative_timings;

    int current_cardinality = initial_card;

    for (int i = 0; i < num_insertion_batches; i++) {
        cout << "\n" << string(60, '=') << endl;
        cout << "Processing batch " << (i + 1) << "/" << num_insertion_batches << " | Edges: " << batches_of_edges[i].size() << endl;

        auto batch_start = chrono::high_resolution_clock::now();
        DetailedTimings batch_timings;
        processBatchGPUVertexClustering(batches_of_edges[i], graph, gpu_graph, hop_limit, batch_timings);
        cumulative_timings.accumulate(batch_timings);

        auto batch_end = chrono::high_resolution_clock::now();
        time_cum += chrono::duration_cast<chrono::microseconds>(batch_end - batch_start).count() / 1000.0;

        // Per-batch cardinality is not read back here to avoid an extra
        // device round trip on every batch; see the final cardinality below.
        cout << "  Current MIS cardinality: [not computed per-batch]" << endl;

        cout << "=== BATCH TIMING BREAKDOWN ===" << endl;
        cout << fixed << setprecision(3);
        cout << "Total Batch Time: " << batch_timings.total_batch_time << " ms\n\n";
        cout << "Component Breakdown:" << endl;
        cout << "  1. CPU Conversion:        " << batch_timings.cpu_conversion_time << " ms (CPU)" << endl;
        cout << "  2. Host->Device Transfer: " << batch_timings.host_to_device_time << " ms (GPU)" << endl;
        cout << "  3. Cluster Processing:    " << batch_timings.cluster_processing_time << " ms (GPU)" << endl;
        cout << "       - Extract vertices: " << batch_timings.extract_vertices_time << " ms" << endl;
        cout << "       - Compute neighb.:  " << batch_timings.compute_neighborhoods_time << " ms" << endl;
        cout << "       - Vertex cluster.:  " << batch_timings.vertex_clustering_time << " ms" << endl;
        cout << "       - Assign edges:     " << batch_timings.assign_edges_time << " ms" << endl;
        cout << "       - Clear assigns.:   " << batch_timings.clear_assignments_time << " ms" << endl;
        cout << "  4. GPU Process:           " << batch_timings.tvb_process_time << " ms (GPU)" << endl;
        cout << "  5. Device->Host Transfer: " << batch_timings.device_to_host_time << " ms (GPU)" << endl;
        cout << "  6. CPU Graph Update:      " << batch_timings.cpu_graph_update_time << " ms (CPU)" << endl;
        cout << "  7. CPU Overhead:          " << batch_timings.cpu_overhead_time << " ms (CPU)" << endl;
        cout << "  8. Cluster count:         " << batch_timings.clusters_count << endl;
        cout << "Finished MIS for batch " << i << " -> slot " << (i % 2) << endl;
        gpu_memory_sample();
    }

    // Synchronize pipeline and get final MIS
    gpu_graph.synchronizeChannels();

    auto d2h_start = chrono::high_resolution_clock::now();
    vector<bool> gpu_mis = gpu_graph.getMISToHost();
    auto d2h_end = chrono::high_resolution_clock::now();
    float final_d2h_time = chrono::duration_cast<chrono::microseconds>(d2h_end - d2h_start).count() / 1000.0;

    for (size_t i = 0; i < initial_mis.size(); i++) {
        initial_mis[i] = gpu_mis[i];
    }

    int updated_card = 0;
    for (int i = 0; i < num_nodes; i++) if (initial_mis[i]) updated_card++;

    // Save the final MIS (1-indexed, one node id per line) for cross-checking
    // against the in-RAM baseline binaries' output.
    {
        string out_dir = "datasets/Final_MISs";
        filesystem::create_directories(out_dir);
        string graph_name = filesystem::path(graph_path).stem().string();
        string final_mis_file = out_dir + "/" + graph_name + "_flash_final_mis.txt";
        ofstream out_file(final_mis_file);
        if (out_file.is_open()) {
            for (int i = 0; i < num_nodes; i++) {
                if (initial_mis[i]) out_file << (i + 1) << "\n";
            }
            out_file.close();
            cout << "Saved final MIS to " << final_mis_file << endl;
        } else {
            cerr << "Warning: Could not open file to save final MIS: " << final_mis_file << endl;
        }
    }

    // MISFIT_SKIP_COMPACTION=1 skips this and leaves the on-disk .egr untouched.
    bool skip_compaction = false;
    {
        const char* skip_env = getenv("MISFIT_SKIP_COMPACTION");
        skip_compaction = (skip_env != nullptr) && (string(skip_env) == "1");
    }

    double compact_time_ms = 0.0;
    if (!skip_compaction) {
        auto compact_start = chrono::high_resolution_clock::now();
        compact_flash_graph(graph.base, graph.delta, graph.flash_path);
        graph.base.remap(graph.flash_path.c_str());
        graph.delta.clear();
        auto compact_end = chrono::high_resolution_clock::now();
        compact_time_ms = chrono::duration_cast<chrono::microseconds>(compact_end - compact_start).count() / 1000.0;
    } else {
        cout << "\n[MISFIT_SKIP_COMPACTION=1] Skipping end-of-cycle compaction — on-disk .egr left untouched." << endl;
    }

    int processed_batches = num_insertion_batches;
    if (processed_batches == 0) processed_batches = 1; // Avoid division by zero

    cout << "\n" << string(80, '=') << endl;
    cout << "=== FINAL RESULTS ===" << endl;
    cout << string(80, '=') << "\n" << endl;

    cout << "Graph Statistics:" << endl;
    cout << "  Initial MIS cardinality: " << initial_card << endl;
    cout << "  Final MIS cardinality:   " << updated_card << endl;
    cout << "  Cardinality change:      " << abs(initial_card - updated_card) << endl;
    cout << "  Number of batches:       " << processed_batches << "\n" << endl;

    cout << "Overall Timing Results:" << endl;
    cout << fixed << setprecision(3);
    cout << "  Total processing time:   " << time_cum << " milliseconds" << endl;
    cout << "  Average time per batch:  " << time_cum / processed_batches << " milliseconds" << endl;
    cout << "  Final D2H Transfer Time: " << final_d2h_time << " milliseconds" << endl;
    cout << setprecision(1);
    cout << "  Processing rate:         " << (processed_batches / (time_cum / 1000.0)) << " batches/second\n" << endl;

    cout << "Average Component Times:" << endl;
    cout << fixed << setprecision(3);
    cout << "  Average Cluster Processing: " << (cumulative_timings.cluster_processing_time / processed_batches) << " ms" << endl;
    cout << "  Average TVB Time:           " << (cumulative_timings.tvb_process_time / processed_batches) << " ms" << endl;
    cout << "\nEnd-of-cycle compaction time (NOT counted in per-batch numbers above): " << compact_time_ms << " ms" << endl;

    gpu_memory_sample();
    double peak_pct = (g_gpu_total_mb > 0.0) ? (g_peak_delta_mb / g_gpu_total_mb * 100.0) : 0.0;
    cout << "Peak program GPU memory: " << setprecision(2) << g_peak_delta_mb << " MB ("
         << setprecision(1) << peak_pct << "% of " << setprecision(2) << g_gpu_total_mb << " MB total)" << endl;

    cout << string(80, '=') << endl;

    return 0;
}
