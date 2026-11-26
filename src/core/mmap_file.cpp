#include "willm/core/mmap_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <system_error>
#include <utility>

namespace willm::core {

namespace {

[[noreturn]] void throw_errno(const std::string& what, const std::filesystem::path& path) {
    throw std::system_error(errno, std::generic_category(), what + " " + path.string());
}

// Closes a file descriptor when it goes out of scope, so every early return
// and every throw below releases it.
class FileDescriptor {
public:
    explicit FileDescriptor(int fd) : fd_(fd) {}
    ~FileDescriptor() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    int get() const { return fd_; }

private:
    int fd_;
};

} // namespace

MmapFile MmapFile::open(const std::filesystem::path& path) {
    FileDescriptor fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.get() < 0) {
        throw_errno("cannot open", path);
    }

    struct stat st {};
    if (::fstat(fd.get(), &st) != 0) {
        throw_errno("cannot stat", path);
    }
    const auto size = static_cast<std::size_t>(st.st_size);

    // mmap rejects a zero length, so an empty file maps to nothing.
    if (size == 0) {
        return MmapFile(nullptr, 0, path);
    }

    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
    if (addr == MAP_FAILED) {
        throw_errno("cannot mmap", path);
    }
    // The mapping keeps its own reference to the file, so the descriptor can
    // be closed here (by FileDescriptor's destructor) while the mapping lives on.
    return MmapFile(static_cast<const std::byte*>(addr), size, path);
}

MmapFile::~MmapFile() {
    reset();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
      path_(std::move(other.path_)) {}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        path_ = std::move(other.path_);
    }
    return *this;
}

void MmapFile::reset() noexcept {
    if (data_ != nullptr) {
        // munmap takes a non const pointer even though we never write.
        ::munmap(const_cast<std::byte*>(data_), size_);
    }
    data_ = nullptr;
    size_ = 0;
    path_.clear();
}

} // namespace willm::core
