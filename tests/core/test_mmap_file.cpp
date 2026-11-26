#include "willm/core/mmap_file.h"

#include <doctest/doctest.h>

#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;
using willm::core::MmapFile;

namespace {

// Writes `contents` to a fresh file in the temp directory and deletes it on scope exit.
class TempFile {
public:
    TempFile(const std::string& name, const std::string& contents)
        : path_(fs::temp_directory_path() / ("willm_" + std::to_string(::getpid()) + "_" + name)) {
        std::ofstream out(path_, std::ios::binary);
        out << contents;
    }
    ~TempFile() {
        std::error_code ec;
        fs::remove(path_, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

} // namespace

TEST_CASE("MmapFile maps a file's bytes") {
    const TempFile tmp("hello.bin", "hello, mmap");
    const auto file = MmapFile::open(tmp.path());

    CHECK(file.size() == 11);
    CHECK_FALSE(file.empty());
    CHECK(file.path() == tmp.path());
    CHECK(std::memcmp(file.data(), "hello, mmap", 11) == 0);
    CHECK(file.bytes().size() == 11);
}

TEST_CASE("MmapFile handles an empty file") {
    const TempFile tmp("empty.bin", "");
    const auto file = MmapFile::open(tmp.path());

    CHECK(file.empty());
    CHECK(file.data() == nullptr);
    CHECK(file.bytes().empty());
}

TEST_CASE("MmapFile throws for a missing file") {
    CHECK_THROWS_AS(MmapFile::open("/definitely/not/a/real/file.bin"), std::system_error);
}

TEST_CASE("MmapFile is move only and transfers ownership") {
    const TempFile tmp("move.bin", "abc");
    auto a = MmapFile::open(tmp.path());
    const std::byte* mapped = a.data();

    MmapFile b = std::move(a);
    CHECK(b.data() == mapped);
    CHECK(b.size() == 3);
    CHECK(a.empty()); // NOLINT(bugprone-use-after-move): moved from state is defined as empty

    MmapFile c;
    c = std::move(b);
    CHECK(c.data() == mapped);
    CHECK(b.empty()); // NOLINT(bugprone-use-after-move)

    static_assert(!std::is_copy_constructible_v<MmapFile>);
    static_assert(!std::is_copy_assignable_v<MmapFile>);
}
