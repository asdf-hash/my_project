// Inventory, click and GUI-batch checks. `xmake build inventory_tests && xmake run inventory_tests`
#include "../src/gui.hpp"
#include "../src/inventory.hpp"
#include "../src/item_entity.hpp"
#include "../src/job_pool.hpp"

#include <chrono>
#include <cmath>
#include <set>
#include <tuple>

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      ++failures;                                                                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                                               \
    }                                                                                                                  \
  } while (0)

ItemStack stack(std::uint16_t block, std::uint16_t count) { return ItemStack{.block = block, .count = count}; }

/** Menu slot id whose inventory index is `index`. */
int slotOf(const inv::Menu &menu, int index) {
  for (int i = 0; i < static_cast<int>(menu.slots().size()); ++i) {
    if (menu.slots()[static_cast<size_t>(i)].index == index) {
      return i;
    }
  }
  return -1;
}

void testAdd() {
  inv::Inventory inventory;
  inventory.selected = 4;
  inventory.items[4] = stack(1, 60);
  inventory.items[2] = stack(1, 10);
  // Selected slot first, then main slots in order, then empty ones hotbar first.
  ItemStack rest = inventory.add(stack(1, 100));
  CHECK(rest.empty());
  CHECK(inventory.items[4].count == 64);
  CHECK(inventory.items[2].count == 64);
  CHECK(inventory.items[0] == stack(1, 42));
  // Offhand merges but is never filled empty.
  inventory.items[inv::kOffhand] = stack(3, 1);
  inventory.add(stack(3, 5));
  CHECK(inventory.items[inv::kOffhand].count == 6);
  // Full inventory returns the remainder.
  inv::Inventory full;
  for (int i = 0; i < inv::kMainSlots; ++i) {
    full.items[static_cast<size_t>(i)] = stack(7, 64);
  }
  full.items[5] = stack(2, 63);
  rest = full.add(stack(2, 10));
  CHECK(rest == stack(2, 9));
  CHECK(full.room(2, 100) == 0);
  CHECK(inventory.room(1, 3000) == 64 * 33 + 22); // 33 empty main slots + 22 room in slot 0
}

void testPickup() {
  inv::Inventory inventory;
  inv::Menu menu(inventory, inv::survivalSlots());
  std::vector<ItemStack> dropped;
  const inv::Menu::DropFn drop = [&](const ItemStack &s) { dropped.push_back(s); };
  const int a = slotOf(menu, 9);
  const int b = slotOf(menu, 10);
  inventory.items[9] = stack(1, 33);
  // Right click takes half (rounded up).
  menu.click(a, 1, inv::ClickType::Pickup, false, drop);
  CHECK(menu.carried == stack(1, 17));
  CHECK(inventory.items[9] == stack(1, 16));
  // Right click on an empty slot places one; left places the rest.
  menu.click(b, 1, inv::ClickType::Pickup, false, drop);
  CHECK(inventory.items[10] == stack(1, 1));
  menu.click(b, 0, inv::ClickType::Pickup, false, drop);
  CHECK(inventory.items[10] == stack(1, 17));
  CHECK(menu.carried.empty());
  // Different item swaps.
  inventory.items[11] = stack(2, 5);
  menu.click(slotOf(menu, 11), 0, inv::ClickType::Pickup, false, drop);
  menu.click(a, 0, inv::ClickType::Pickup, false, drop);
  CHECK(inventory.items[9] == stack(2, 5));
  CHECK(menu.carried == stack(1, 16));
  // Merge caps at 64.
  inventory.items[12] = stack(1, 60);
  menu.click(slotOf(menu, 12), 0, inv::ClickType::Pickup, false, drop);
  CHECK(inventory.items[12].count == 64);
  CHECK(menu.carried == stack(1, 12));
  // Outside: right drops one, left drops all.
  menu.click(inv::Menu::kOutside, 1, inv::ClickType::Pickup, false, drop);
  CHECK(dropped.size() == 1 && dropped[0] == stack(1, 1));
  menu.click(inv::Menu::kOutside, 0, inv::ClickType::Pickup, false, drop);
  CHECK(dropped.size() == 2 && dropped[1] == stack(1, 11));
  CHECK(menu.carried.empty());
  // Armor takes nothing.
  menu.carried = stack(1, 3);
  menu.click(slotOf(menu, inv::kArmorBase), 0, inv::ClickType::Pickup, false, drop);
  CHECK(inventory.items[inv::kArmorBase].empty());
  CHECK(menu.carried == stack(1, 3));
}

void testQuickMoveSwapThrow() {
  inv::Inventory inventory;
  inv::Menu menu(inventory, inv::survivalSlots());
  std::vector<ItemStack> dropped;
  const inv::Menu::DropFn drop = [&](const ItemStack &s) { dropped.push_back(s); };
  // Storage -> hotbar, merging first.
  inventory.items[3] = stack(1, 60);
  inventory.items[20] = stack(1, 10);
  menu.click(slotOf(menu, 20), 0, inv::ClickType::QuickMove, false, drop);
  CHECK(inventory.items[3].count == 64);
  CHECK(inventory.items[0] == stack(1, 6));
  CHECK(inventory.items[20].empty());
  // Hotbar -> storage.
  menu.click(slotOf(menu, 0), 0, inv::ClickType::QuickMove, false, drop);
  CHECK(inventory.items[0].empty());
  CHECK(inventory.items[9] == stack(1, 6));
  // Offhand -> storage.
  inventory.items[inv::kOffhand] = stack(2, 4);
  menu.click(slotOf(menu, inv::kOffhand), 0, inv::ClickType::QuickMove, false, drop);
  CHECK(inventory.items[inv::kOffhand].empty());
  CHECK(inventory.items[10] == stack(2, 4));
  // Number key swaps with the hotbar; F with the offhand.
  menu.click(slotOf(menu, 10), 5, inv::ClickType::Swap, false, drop);
  CHECK(inventory.items[5] == stack(2, 4));
  CHECK(inventory.items[10].empty());
  menu.click(slotOf(menu, 9), inv::kOffhand, inv::ClickType::Swap, false, drop);
  CHECK(inventory.items[inv::kOffhand] == stack(1, 6));
  // Q drops one, Ctrl+Q the stack.
  menu.click(slotOf(menu, 5), 0, inv::ClickType::Throw, false, drop);
  CHECK(dropped.back() == stack(2, 1));
  menu.click(slotOf(menu, 5), 1, inv::ClickType::Throw, false, drop);
  CHECK(dropped.back() == stack(2, 3));
  CHECK(inventory.items[5].empty());
  // Clone only in creative.
  menu.click(slotOf(menu, 3), 2, inv::ClickType::Clone, false, drop);
  CHECK(menu.carried.empty());
  inventory.items[4] = stack(1, 2);
  menu.click(slotOf(menu, 4), 2, inv::ClickType::Clone, true, drop);
  CHECK(menu.carried == stack(1, 64));
}

void testPickupAllAndDrag() {
  inv::Inventory inventory;
  inv::Menu menu(inventory, inv::survivalSlots());
  const inv::Menu::DropFn drop = [](const ItemStack &) {};
  inventory.items[9] = stack(1, 64);
  inventory.items[10] = stack(1, 5);
  inventory.items[11] = stack(1, 7);
  menu.carried = stack(1, 2);
  // Double-click: partial stacks first, then full ones.
  menu.click(slotOf(menu, 12), 0, inv::ClickType::PickupAll, false, drop);
  CHECK(menu.carried.count == 64);
  CHECK(inventory.items[10].empty() && inventory.items[11].empty());
  CHECK(inventory.items[9].count == 14);
  // Left drag splits evenly, the remainder stays carried.
  menu.carried = stack(2, 10);
  menu.beginDrag(0);
  menu.dragOver(slotOf(menu, 20));
  menu.dragOver(slotOf(menu, 21));
  menu.dragOver(slotOf(menu, 22));
  menu.dragOver(slotOf(menu, 22)); // twice: ignored
  menu.endDrag(false, drop);
  CHECK(inventory.items[20] == stack(2, 3) && inventory.items[21] == stack(2, 3) && inventory.items[22] == stack(2, 3));
  CHECK(menu.carried == stack(2, 1));
  // Right drag places one each, skipping slots of another item.
  menu.carried = stack(2, 5);
  menu.beginDrag(1);
  menu.dragOver(slotOf(menu, 9)); // block 1: refused
  menu.dragOver(slotOf(menu, 23));
  menu.dragOver(slotOf(menu, 24));
  menu.endDrag(false, drop);
  CHECK(inventory.items[9] == stack(1, 14));
  CHECK(inventory.items[23] == stack(2, 1) && inventory.items[24] == stack(2, 1));
  CHECK(menu.carried == stack(2, 3));
  // A drag over a single slot is an ordinary click.
  menu.beginDrag(0);
  menu.dragOver(slotOf(menu, 25));
  menu.endDrag(false, drop);
  CHECK(inventory.items[25] == stack(2, 3));
  CHECK(menu.carried.empty());
}

void testCreativeGrid() {
  inv::Inventory inventory;
  inv::Menu menu(inventory, inv::creativeHotbarSlots());
  std::vector<ItemStack> dropped;
  const inv::Menu::DropFn drop = [&](const ItemStack &s) { dropped.push_back(s); };
  const ItemStack grass = stack(0, 1);
  inv::creativeGridClick(menu, grass, 0, inv::ClickType::Pickup, false, drop);
  CHECK(menu.carried == stack(0, 1));
  inv::creativeGridClick(menu, grass, 0, inv::ClickType::Pickup, false, drop);
  CHECK(menu.carried == stack(0, 2));
  inv::creativeGridClick(menu, grass, 0, inv::ClickType::Pickup, true, drop);
  CHECK(menu.carried == stack(0, 64));
  inv::creativeGridClick(menu, grass, 1, inv::ClickType::Pickup, false, drop);
  CHECK(menu.carried == stack(0, 63));
  // Left click on another item (or empty grid) deletes the carried stack.
  inv::creativeGridClick(menu, stack(5, 1), 0, inv::ClickType::Pickup, false, drop);
  CHECK(menu.carried.empty());
  inv::creativeGridClick(menu, grass, 3, inv::ClickType::Swap, false, drop);
  CHECK(inventory.items[3] == stack(0, 64));
  inv::creativeGridClick(menu, grass, 1, inv::ClickType::Throw, false, drop);
  CHECK(dropped.back() == stack(0, 64));
}

void testGuiBatch() {
  gui::Batch batch;
  batch.begin(1920, 1080, gui::guiScale(1080));
  CHECK(batch.scale() == 3);
  CHECK(batch.width() == 640 && batch.height() == 360);
  gui::Sprite sprite{.texture = 7, .width = 100, .height = 100, .scaling = gui::Scaling::NineSlice, .border = 9};
  batch.sprite(sprite, 10, 10, 50, 30);
  CHECK(batch.quads().size() == 9);
  CHECK(batch.quads()[0].rect == glm::vec4(30, 30, 27, 27));
  const gui::Quad &center = batch.quads()[8];
  CHECK(center.source == glm::vec4(9, 9, 82, 82) && center.extent.x == 32.0f && center.extent.y == 12.0f);
  CHECK(gui::packArgb(0xff112233u) == 0xff332211u);
}

// Fake world: solid below y = 0, extra blocks in `solid`, water where y < waterTop inside the water box.
struct FakeWorld : collision::CellQuery {
  std::set<std::tuple<int, int, int>> solid;
  std::set<std::tuple<int, int, int>> removed;
  int waterMinX = 1000, waterMaxX = 1000, waterTop = 0;
  bool collidable(std::int32_t x, std::int32_t y, std::int32_t z) const override {
    if (removed.contains({x, y, z})) {
      return false;
    }
    return y < 0 || solid.contains({x, y, z});
  }
  bool water(std::int32_t x, std::int32_t y, std::int32_t z) const override {
    (void)z;
    return x >= waterMinX && x <= waterMaxX && y >= 0 && y < waterTop && !collidable(x, y, z);
  }
  bool loaded(std::int32_t, std::int32_t) const override { return true; }
};

void tickN(ItemEntityWorld &items, const FakeWorld &world, int ticks, JobPool *pool = nullptr,
           const ItemEntityWorld::Collector *collector = nullptr) {
  for (int t = 0; t < ticks; ++t) {
    items.tick(world, pool, collector);
  }
}

constexpr double kDirt = 1300.0;
constexpr double kLog = 700.0;
constexpr double kStone = 2600.0;

/** Spawn a body at rest (no spin). */
void drop(ItemEntityWorld &items, const glm::dvec3 &center, const glm::dvec3 &velocity, ItemStack item, double density,
          int delay = 0, std::uint32_t seed = 1, int mergeDelay = 0) {
  items.spawn(center, velocity, glm::dvec3(0.0), density, item, delay, seed, mergeDelay);
}

void testItemPhysics() {
  FakeWorld world;
  // Falls, lands on a face (center a half size above the floor) and sleeps.
  {
    ItemEntityWorld items;
    drop(items, {0.5, 5.0, 0.5}, {0.0, 0.0, 0.0}, stack(1, 1), kDirt);
    tickN(items, world, 120);
    CHECK(items.size() == 1);
    CHECK(std::abs(items.position[0].y - ItemEntityWorld::kHalfSize) < 0.01);
    CHECK((items.flags[0] & ItemEntityWorld::kAsleep) != 0);
    // Resting flat: the body's up axis lines up with a world axis.
    const glm::dmat3 r = glm::mat3_cast(items.orientation[0]);
    const double maxUp = std::max({std::abs(r[0].y), std::abs(r[1].y), std::abs(r[2].y)});
    CHECK(maxUp > 0.99);
    // Wakes and falls when the floor under it goes away (it may have tumbled away from the drop spot).
    const int hx = static_cast<int>(std::floor(items.position[0].x));
    const int hz = static_cast<int>(std::floor(items.position[0].z));
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dz = -1; dz <= 1; ++dz) {
        world.removed.insert({hx + dx, -1, hz + dz});
      }
    }
    items.wakeAround(hx, -1, hz);
    tickN(items, world, 60);
    CHECK(std::abs(items.position[0].y - (-1.0 + ItemEntityWorld::kHalfSize)) < 0.02);
    world.removed.clear();
  }
  // Thrown hard at a wall: stops against it, no tunnelling.
  {
    FakeWorld walled = world;
    for (int y = 0; y < 4; ++y) {
      walled.solid.insert({3, y, 0});
    }
    ItemEntityWorld items;
    drop(items, {0.5, 0.5, 0.5}, {60.0, 0.0, 0.0}, stack(1, 1), kDirt);
    tickN(items, walled, 60);
    const collision::Box box = items.bounds(0);
    CHECK(box.max.x <= 3.0 + 0.01);
    CHECK(items.position[0].x < 3.0 && items.position[0].x > -40.0); // bounced back (restitution), never through
  }
  // Tumbles: dropped spinning onto a slope of steps, it comes to rest without sinking into anything.
  {
    FakeWorld steps = world;
    for (int x = 0; x < 4; ++x) {
      for (int y = 0; y < 4 - x; ++y) {
        for (int z = -2; z <= 2; ++z) {
          steps.solid.insert({x, y, z});
        }
      }
    }
    ItemEntityWorld items;
    items.spawn({0.5, 6.0, 0.5}, {2.0, 0.0, 0.0}, {3.0, 1.0, -2.0}, kDirt, stack(1, 1), 0, 7);
    tickN(items, steps, 200);
    const glm::dvec3 p = items.position[0];
    const glm::dmat3 r = glm::mat3_cast(items.orientation[0]);
    bool buried = false;
    for (int x = -1; x <= 1; x += 2) {
      for (int y = -1; y <= 1; y += 2) {
        for (int z = -1; z <= 1; z += 2) {
          const glm::dvec3 corner = p + r * (glm::dvec3(x, y, z) * (ItemEntityWorld::kHalfSize - 0.01));
          buried = buried || steps.collidable(static_cast<int>(std::floor(corner.x)), static_cast<int>(std::floor(corner.y)),
                                              static_cast<int>(std::floor(corner.z)));
        }
      }
    }
    CHECK(!buried);
    CHECK((items.flags[0] & ItemEntityWorld::kOnGround) != 0);
  }
  // Pushed out of a block placed on it.
  {
    FakeWorld boxed = world;
    for (const auto &cell : {std::tuple{0, 0, 0}, std::tuple{1, 0, 0}, std::tuple{-1, 0, 0}, std::tuple{0, 0, 1},
                             std::tuple{0, 0, -1}}) {
      boxed.solid.insert(cell);
    }
    ItemEntityWorld items;
    drop(items, {0.5, 0.3, 0.5}, {0.0, 0.0, 0.0}, stack(1, 1), kDirt);
    tickN(items, boxed, 100);
    const glm::dvec3 p = items.position[0];
    CHECK(!boxed.collidable(static_cast<int>(std::floor(p.x)), static_cast<int>(std::floor(p.y)),
                            static_cast<int>(std::floor(p.z))));
    CHECK((items.flags[0] & ItemEntityWorld::kOnGround) != 0);
  }
  // Density: a log floats with ~70% under the surface, stone sinks to the bottom.
  {
    FakeWorld lake = world;
    lake.waterMinX = -5;
    lake.waterMaxX = 5;
    lake.waterTop = 6;
    ItemEntityWorld items;
    drop(items, {0.5, 1.0, 0.5}, {0.0, 0.0, 0.0}, stack(1, 1), kLog, 0, 1);
    drop(items, {2.5, 5.5, 0.5}, {0.0, 0.0, 0.0}, stack(2, 1), kStone, 0, 2);
    tickN(items, lake, 600);
    const std::size_t log = items.stack[0].block == 1 ? 0 : 1;
    const std::size_t stone = 1 - log;
    CHECK(items.position[log].y > 5.8 && items.position[log].y < 6.05);
    CHECK((items.flags[log] & ItemEntityWorld::kInWater) != 0);
    CHECK(std::abs(items.position[stone].y - ItemEntityWorld::kHalfSize) < 0.02);
  }
  // Thrown items cannot merge until their merge delay expires.
  {
    ItemEntityWorld items;
    drop(items, {0.5, 0.125, 0.5}, {0.0, 0.0, 0.0}, stack(2, 10), kDirt, 40, 20,
         ItemEntityWorld::kThrowMergeDelay);
    drop(items, {0.5, 0.125, 0.5}, {0.0, 0.0, 0.0}, stack(2, 15), kDirt, 40, 21,
         ItemEntityWorld::kThrowMergeDelay);
    tickN(items, world, 39);
    CHECK(items.size() == 2);
    tickN(items, world, 1);
    CHECK(items.size() == 1);
    CHECK(items.stack[0].count == 25);
  }
  // Merging caps stacks at 2048.
  {
    ItemEntityWorld items;
    for (int i = 0; i < 5; ++i) {
      drop(items, {0.5 + 0.01 * i, 0.125, 0.5}, {0.0, 0.0, 0.0}, stack(2, 1000), kDirt, 0, static_cast<std::uint32_t>(10 + i));
    }
    drop(items, {0.6, 0.125, 0.6}, {0.0, 0.0, 0.0}, stack(3, 5), kDirt, 0, 20); // another block: never merged
    tickN(items, world, 100);
    int total = 0;
    int maxCount = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (items.stack[i].block == 2) {
        total += items.stack[i].count;
        maxCount = std::max<int>(maxCount, items.stack[i].count);
      }
    }
    CHECK(total == 5000);
    CHECK(maxCount == 2048);
    CHECK(items.size() == 4); // 2048 + 2048 + 904, and the other block
    // Each fully absorbed body flies into the stack that took it.
    int flown = 0;
    for (const ItemEntityWorld::Flight &flight : items.takeFlights()) {
      CHECK(!flight.toCollector && flight.stack.block == 2);
      flown += flight.stack.count;
    }
    CHECK(flown > 0);
  }
  // Pickup takes what fits and leaves the rest; pickup delay is honored.
  {
    ItemEntityWorld items;
    drop(items, {0.5, 0.125, 0.5}, {0.0, 0.0, 0.0}, stack(4, 100), kDirt, 15, 30);
    int collected = 0;
    const ItemEntityWorld::Collector player{.box = {{0.2, 0.0, 0.2}, {0.8, 1.8, 0.8}},
                                            .take = [&](const ItemStack &s) {
                                              const int taken = std::min<int>(s.count, 64 - collected);
                                              collected += taken;
                                              return taken;
                                            }};
    tickN(items, world, 10, nullptr, &player);
    CHECK(collected == 0);
    tickN(items, world, 10, nullptr, &player);
    CHECK(collected == 64);
    CHECK(items.size() == 1 && items.stack[0].count == 36);
    // The 64 taken fly to the player (the renderer animates them).
    const std::vector<ItemEntityWorld::Flight> flights = items.takeFlights();
    CHECK(flights.size() == 1 && flights[0].toCollector && flights[0].stack == stack(4, 64));
    CHECK(items.takeFlights().empty());
  }
  // Despawn after 5 minutes.
  {
    ItemEntityWorld items;
    drop(items, {0.5, 0.125, 0.5}, {0.0, 0.0, 0.0}, stack(1, 1), kDirt);
    tickN(items, world, ItemEntityWorld::kLifetimeTicks - 1);
    CHECK(items.size() == 1);
    tickN(items, world, 1);
    CHECK(items.size() == 0);
  }
}

void testParallelMatchesSerial() {
  FakeWorld world;
  for (int x = -20; x <= 20; x += 3) {
    world.solid.insert({x, 0, 2});
  }
  ItemEntityWorld serial;
  for (int i = 0; i < 3000; ++i) {
    const double a = i * 0.37;
    serial.spawn({std::cos(a) * 10.0, 3.0 + (i % 7), std::sin(a) * 10.0},
                 {std::sin(a * 3.0) * 6.0, 4.0, std::cos(a * 5.0) * 6.0}, {std::sin(a), std::cos(a), 0.5},
                 i % 2 == 0 ? kDirt : kLog, stack(static_cast<std::uint16_t>(i % 3), 1), 10, static_cast<std::uint32_t>(i));
  }
  ItemEntityWorld parallel = serial;
  JobPool pool(3);
  const auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < 120; ++t) {
    parallel.tick(world, &pool, nullptr);
  }
  const double parallelMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const auto serialStart = std::chrono::steady_clock::now();
  for (int t = 0; t < 120; ++t) {
    serial.tick(world, nullptr, nullptr);
  }
  const double serialMs =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - serialStart).count();
  std::printf("item physics, 3000 bodies x 120 ticks: serial %.1f ms, 3 workers + main %.1f ms\n", serialMs, parallelMs);
  CHECK(serial.size() == parallel.size());
  bool same = serial.size() == parallel.size();
  for (std::size_t i = 0; same && i < serial.size(); ++i) {
    same = serial.position[i] == parallel.position[i] && serial.stack[i] == parallel.stack[i];
  }
  CHECK(same);
}

} // namespace

int main() {
  testAdd();
  testPickup();
  testQuickMoveSwapThrow();
  testPickupAllAndDrag();
  testCreativeGrid();
  testGuiBatch();
  testItemPhysics();
  testParallelMatchesSerial();
  std::printf(failures == 0 ? "ALL PASSED\n" : "%d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
