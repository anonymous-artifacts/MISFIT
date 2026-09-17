// core/reporting.h
//
// Standardized console output shared by both MISFIT binaries.

#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

struct ReportKV {
    std::string key;
    std::string value;
};

inline void print_run_header(const std::string& algo_name, const std::string& graph_path,
                              long long num_nodes, long long num_edges, double avg_degree,
                              const std::string& gpu_name, bool integrated) {
    std::cout << "=== MISFIT " << algo_name << " ===" << std::endl;
    std::cout << "Graph: " << graph_path << std::endl;
    std::cout << "  nodes=" << num_nodes << " edges=" << num_edges
              << " avg_degree=" << std::fixed << std::setprecision(2) << avg_degree << std::endl;
    std::cout << "GPU: " << gpu_name << " | "
              << (integrated ? "integrated (zero-copy path)" : "discrete (explicit-transfer path)")
              << std::endl;
}

inline void print_batch_result(int batch_idx, int num_batches, int edges_in_batch, double batch_ms,
                                const std::vector<ReportKV>& extra = {}) {
    std::cout << "Batch " << batch_idx << "/" << num_batches
              << " | edges=" << edges_in_batch
              << " | time=" << std::fixed << std::setprecision(3) << batch_ms << "ms";
    for (const auto& kv : extra) std::cout << " | " << kv.key << "=" << kv.value;
    std::cout << std::endl;
}

inline void print_final_summary(double setup_ms, double processing_ms, double e2e_ms,
                                 const std::string& initial_label, long long initial_value,
                                 const std::string& final_label, long long final_value) {
    std::cout << "\n=== FINAL SUMMARY ===" << std::endl;
    std::cout << "Setup time:      " << std::fixed << std::setprecision(3) << setup_ms << " ms" << std::endl;
    std::cout << "Processing time: " << std::fixed << std::setprecision(3) << processing_ms << " ms" << std::endl;
    std::cout << "End-to-end time: " << std::fixed << std::setprecision(3) << e2e_ms << " ms" << std::endl;
    std::cout << "Initial " << initial_label << ": " << initial_value << std::endl;
    std::cout << "Final " << final_label << ":   " << final_value << std::endl;
}
