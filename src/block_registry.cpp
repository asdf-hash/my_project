#include "block_registry.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace blocks {
namespace {

using FaceMask = std::uint32_t; // bit i = face i

constexpr FaceMask kAllFaces = 0b111111u;

std::optional<FaceMask> parseFaceSelector(std::string_view selector) {
  static constexpr std::pair<std::string_view, FaceMask> selectors[] = {
      {"all", kAllFaces},       {"side", 0b110011u},      {"top", 1u << 2},         {"bottom", 1u << 3},
      {"positive_x", 1u << 0},  {"negative_x", 1u << 1},  {"positive_y", 1u << 2},  {"negative_y", 1u << 3},
      {"positive_z", 1u << 4},  {"negative_z", 1u << 5},
  };
  for (const auto &[name, mask] : selectors) {
    if (name == selector) {
      return mask;
    }
  }
  return std::nullopt;
}

std::string_view trim(std::string_view value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string_view> splitWords(std::string_view value) {
  std::vector<std::string_view> words;
  while (!(value = trim(value)).empty()) {
    const auto end = value.find_first_of(" \t");
    words.push_back(value.substr(0, end));
    value = end == std::string_view::npos ? std::string_view{} : value.substr(end);
  }
  return words;
}

class Parser {
public:
  Parser(std::string path, std::string textureDirectory)
      : path_(std::move(path)), textureDirectory_(std::move(textureDirectory)) {}

  Registry parse() {
    std::ifstream file(path_);
    if (!file) {
      throw std::runtime_error("failed to open block definitions: " + path_);
    }

    std::string line;
    while (std::getline(file, line)) {
      ++lineNumber_;
      std::string_view content = line;
      content = trim(content.substr(0, content.find('#')));
      if (content.empty()) {
        continue;
      }
      if (content.front() == '[') {
        beginBlock(content);
      } else {
        parseProperty(content);
      }
    }
    finishBlock();

    if (registry_.blocks.empty()) {
      throw std::runtime_error("no blocks defined in " + path_);
    }
    return std::move(registry_);
  }

private:
  [[noreturn]] void fail(const std::string &message) const {
    throw std::runtime_error(path_ + ":" + std::to_string(lineNumber_) + ": " + message);
  }

  void beginBlock(std::string_view header) {
    finishBlock();
    if (header.back() != ']') {
      fail("expected [block_name]");
    }
    const std::string name(trim(header.substr(1, header.size() - 2)));
    if (name.empty()) {
      fail("empty block name");
    }
    if (std::ranges::any_of(registry_.blocks, [&](const Block &block) { return block.name == name; })) {
      fail("duplicate block '" + name + "'");
    }
    registry_.blocks.push_back({.name = name});
    definedFaces_ = 0;
    hasLodTint_ = false;
    blockLine_ = lineNumber_;
  }

  void finishBlock() {
    if (registry_.blocks.empty()) {
      return;
    }
    Block &block = registry_.blocks.back();
    const auto blockError = [&](const std::string &message) {
      return std::runtime_error(path_ + ":" + std::to_string(blockLine_) + ": " + message);
    };
    if (block.opaque) {
      if (opaqueCount_ >= kMaxOpaqueBlocks) {
        throw blockError("too many opaque blocks (max " + std::to_string(kMaxOpaqueBlocks) + ")");
      }
      block.id = static_cast<std::uint32_t>(opaqueCount_++);
    } else {
      if (nonOpaqueCount_ >= kMaxNonOpaqueBlocks) {
        throw blockError("too many non-opaque blocks (max " + std::to_string(kMaxNonOpaqueBlocks) + ")");
      }
      block.id = kNonOpaqueBit + static_cast<std::uint32_t>(nonOpaqueCount_++);
    }
    if (block.fluid != Fluid::None && block.opaque) {
      throw blockError("block '" + block.name + "' is a fluid, so it must set opaque = false");
    }
    if (definedFaces_ != kAllFaces) {
      throw std::runtime_error(path_ + ":" + std::to_string(blockLine_) + ": block '" + block.name +
                               "' must give every face a texture (use all/side/top/bottom)");
    }
    if (hasLodTint_ && std::ranges::none_of(block.faces, [](const Face &face) { return face.hasLodTexture; })) {
      throw std::runtime_error(path_ + ":" + std::to_string(blockLine_) + ": block '" + block.name +
                               "' sets lod_tint without any lod_<face> texture");
    }
  }

  void parseProperty(std::string_view content) {
    if (registry_.blocks.empty()) {
      fail("property outside of a [block] section");
    }
    const auto separator = content.find('=');
    if (separator == std::string_view::npos) {
      fail("expected key = value");
    }
    const std::string_view key = trim(content.substr(0, separator));
    const std::string_view value = trim(content.substr(separator + 1));
    if (value.empty()) {
      fail("missing value for '" + std::string(key) + "'");
    }
    Block &block = registry_.blocks.back();

    if (const auto faces = parseFaceSelector(key)) {
      const std::uint32_t texture = textureIndex(value);
      forEachFace(*faces, block, [&](Face &face) { face.texture = texture; });
      definedFaces_ |= *faces;
    } else if (key.starts_with("lod_") && parseFaceSelector(key.substr(4))) {
      const std::uint32_t texture = textureIndex(value);
      forEachFace(*parseFaceSelector(key.substr(4)), block, [&](Face &face) {
        face.lodTexture = texture;
        face.hasLodTexture = true;
      });
    } else if (key == "lod_tint") {
      parseColor(key, value, block.lodTint);
      hasLodTint_ = true;
    } else if (key == "frame_time") {
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), block.frameTime);
      if (error != std::errc{} || end != value.data() + value.size() || !(block.frameTime > 0.0f)) {
        fail("frame_time expects a positive number of seconds");
      }
    } else if (key == "tint") {
      parseColor(key, value, block.tint);
    } else if (key == "fluid") {
      if (value == "none") {
        block.fluid = Fluid::None;
      } else if (value == "ocean") {
        block.fluid = Fluid::Ocean;
      } else {
        fail("fluid expects none or ocean");
      }
    } else if (key == "opaque") {
      if (value == "true") {
        block.opaque = true;
      } else if (value == "false") {
        block.opaque = false;
      } else {
        fail("opaque expects true or false");
      }
    } else if (key == "random_rotation") {
      for (const std::string_view word : splitWords(value)) {
        const auto faces = parseFaceSelector(word);
        if (!faces) {
          fail("unknown face '" + std::string(word) + "' in random_rotation");
        }
        forEachFace(*faces, block, [](Face &face) { face.randomRotation = true; });
      }
    } else {
      fail("unknown key '" + std::string(key) + "'");
    }
  }

  void parseColor(std::string_view key, std::string_view value, std::array<float, 3> &color) const {
    const auto words = splitWords(value);
    if (words.size() != 3) {
      fail(std::string(key) + " expects three numbers (r g b)");
    }
    for (std::size_t i = 0; i < 3; ++i) {
      const auto [end, error] = std::from_chars(words[i].data(), words[i].data() + words[i].size(), color[i]);
      if (error != std::errc{} || end != words[i].data() + words[i].size()) {
        fail("invalid number '" + std::string(words[i]) + "' in " + std::string(key));
      }
    }
  }

  template <typename Fn> static void forEachFace(FaceMask faces, Block &block, Fn &&fn) {
    for (std::size_t i = 0; i < kFaceCount; ++i) {
      if (faces & (1u << i)) {
        fn(block.faces[i]);
      }
    }
  }

  std::uint32_t textureIndex(std::string_view name) {
    const std::string texturePath = textureDirectory_ + std::string(name);
    const auto [it, inserted] =
        textureIndices_.emplace(texturePath, static_cast<std::uint32_t>(registry_.texturePaths.size()));
    if (inserted) {
      registry_.texturePaths.push_back(texturePath);
    }
    return it->second;
  }

  std::string path_;
  std::string textureDirectory_;
  Registry registry_;
  std::unordered_map<std::string, std::uint32_t> textureIndices_;
  std::size_t lineNumber_ = 0;
  std::size_t blockLine_ = 0;
  FaceMask definedFaces_ = 0;
  bool hasLodTint_ = false;
  std::size_t opaqueCount_ = 0;
  std::size_t nonOpaqueCount_ = 0;
};

} // namespace

std::uint32_t Registry::idOf(std::string_view name) const {
  const auto it = std::ranges::find_if(blocks, [&](const Block &block) { return block.name == name; });
  if (it == blocks.end()) {
    throw std::runtime_error("unknown block '" + std::string(name) + "'");
  }
  return it->id;
}

std::uint32_t Registry::maxId() const {
  std::uint32_t id = 0;
  for (const Block &block : blocks) {
    id = std::max(id, block.id);
  }
  return id;
}

Registry loadRegistry(const std::string &path, const std::string &textureDirectory) {
  return Parser(path, textureDirectory).parse();
}

} // namespace blocks
