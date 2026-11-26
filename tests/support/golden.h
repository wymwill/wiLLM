#pragma once

// Reader for the golden activation files written by tools/llama2c.py
// (write_golden). Tensors are viewed in place inside the mapped file.

#include "willm/core/mmap_file.h"
#include "willm/core/tensor_view.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace willm::testing {

class GoldenFile {
public:
    // Throws std::system_error if the file is missing, core::CheckError if it is malformed.
    static GoldenFile open(const std::filesystem::path& path);

    bool contains(const std::string& name) const { return tensors_.contains(name); }
    // Throws core::CheckError for an unknown name.
    core::ConstTensorView get(const std::string& name) const;
    // Names in file order (the order the reference recorded them).
    const std::vector<std::string>& names() const { return names_; }

private:
    core::MmapFile file_;
    std::vector<std::string> names_;
    std::map<std::string, core::ConstTensorView> tensors_;
};

} // namespace willm::testing
