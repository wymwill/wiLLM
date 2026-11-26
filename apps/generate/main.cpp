// generate: prompt in, text out.
//
// M1 stub. It loads a llama2.c checkpoint and prints its config. Tokenizer,
// sampling and the generation loop arrive in M2.

#include "willm/core/timer.h"
#include "willm/model/weights.h"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <checkpoint.bin>\n", argv[0]);
        return 2;
    }
    try {
        const willm::core::Timer timer;
        const auto model = willm::model::load_llama2c(argv[1]);
        const auto& c = model.config;
        std::printf("loaded %s (%.1f MB) in %.3f ms\n", model.file.path().c_str(),
                    static_cast<double>(model.file.size()) / 1e6, timer.elapsed_ms());
        std::printf("dim %d, hidden %d, layers %d, heads %d, kv heads %d, vocab %d, seq_len %d\n",
                    c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size,
                    c.seq_len);
        std::printf("generation is not implemented yet (M2)\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
