// generate: prompt in, text out.
//
// M0 stub. It maps the checkpoint to prove the build and core/ work end to
// end. The forward pass arrives in M1, tokenizer and sampling in M2.

#include "willm/core/mmap_file.h"
#include "willm/core/timer.h"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <checkpoint.bin>\n", argv[0]);
        return 2;
    }
    try {
        const willm::core::Timer timer;
        const auto file = willm::core::MmapFile::open(argv[1]);
        std::printf("mapped %s: %zu bytes in %.3f ms\n", file.path().c_str(), file.size(),
                    timer.elapsed_ms());
        std::printf("generation is not implemented yet (M1, M2)\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
