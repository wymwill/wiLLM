#pragma once

// MmapFile: a read only memory mapping of a whole file, released automatically.
//
// Why mmap instead of reading the file into a vector? Model weights are large
// (60 MB for stories15M, many GB later). With mmap the OS pages weights in on
// first touch, shares them between processes, and lets us point tensor views
// straight into the file with zero copies. On Apple silicon the same pages can
// later be wrapped as Metal buffers without copying either.

#include <cstddef>
#include <filesystem>
#include <span>

namespace willm::core {

class MmapFile {
public:
    // An empty mapping (no file). Useful as a moved from or default state.
    MmapFile() = default;

    // Maps the whole file at `path` read only.
    // Throws std::system_error if the file cannot be opened, stat'ed or mapped.
    // An empty file is valid and gives an empty span.
    static MmapFile open(const std::filesystem::path& path);

    ~MmapFile();

    // Move only. Copying would mean two owners unmapping the same pages.
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }
    const std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    MmapFile(const std::byte* data, std::size_t size, std::filesystem::path path)
        : data_(data), size_(size), path_(std::move(path)) {}

    void reset() noexcept; // unmaps and returns to the empty state

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
    std::filesystem::path path_;
};

} // namespace willm::core
