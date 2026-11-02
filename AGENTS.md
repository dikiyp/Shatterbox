# AGENTS.md

Правила для ИИ-агентов, работающих в `E:\vox_engine`.

## 0. Что это за проект

Собственный воксельный движок **Shatterbox** для тактического арена-шутера в духе Teardown: полная разрушаемость укрытий, динамический свет, инструменты, огонь/дым, матчи 5×5 с ботами. PC-first: **Windows + Linux, без консолей**. Масштаб строго — арена **256×256 м** (высота 64 м). Открытый мир 4–8 км, streaming и floating origin — вне текущего скоупа.

Сейчас идёт **Фаза 0/1**: каркас движка уже собран (workspace, окно, wgpu, egui, ввод, free-cam), дальше воксели и DDA. Единственный источник истины по архитектуре — `docs/ENGINE_PLAN.md` (v0.3). Если код и план расходятся, сначала обновляем план и фиксируем причину, потом код.

## 1. Язык и стиль общения

- Пользователь пишет по-русски; отвечай по-русски, неформально, но по делу.
- Не пересказывай очевидное; сначала результат, потом детали.
- Не обещай 144 fps в любой сцене. Обязательная планка — **стабильные 70–80 fps** в типовых и тяжёлых сценах; 70–80 — не лимит, больше всегда можно; 144 — верхняя цель, не требование; 60 — нижняя граница.
- Если задача огромная — предложи вертикальный срез, а не «сделаем весь движок за раз».

## 2. Границы работы

- Пиши код только в рамках поставленной задачи; не расширяй скоуп и не рефакторь соседние модули без запроса. Если задача больше вертикального среза — сначала предложи срез.
- Не добавлять консоли, платформы, сетевой слой или «как в Unreal» без запроса.
- Не удалять и не перезаписывать пользовательские файлы без необходимости; сомнительные изменения — сначала обсудить.
- Репозиторий не инициализирован как git. Не делать `git init`, коммиты и push без просьбы.
- Не добавлять тяжёлые зависимости без обоснования. Сначала проверь, что в стеке уже есть решение.

## 2а. Среда

- Основная ОС: Windows, PowerShell. Команды давай для PowerShell; пути — `E:\vox_engine\...`.
- Агент сам запускает игру, тесты и замеры через автотест-харнесс (скилл `voxel-autotest`, `docs/tasks/TASK_autotest.md`): сценарий → PNG + report.json + лог. Фразы в `docs/tasks/*` вроде «замеры запускает пользователь», «дай команды для моего запуска», «цифры из моего вывода» отменены этим пунктом: запускай и проверяй сам. Всегда с таймаутом и закрытием процесса; цифры — только из report.json и логов, не из памяти. Шум замеров до ±30 %: A/B в одной сессии, 3 прогона, p50. Пока харнесс не готов, запускай `cargo run --release` с таймаутом и читай лог. Если чего-то сделать не можешь (нет GPU, окна, просмотра изображений) — прямо скажи, что именно не проверено.
- Версии крейтов — из корневого `Cargo.toml`, не из памяти и не из примеров в скиллах.

## 3. Целевые показатели (не менять без пользователя)

- GPU: **RTX 3050 8GB**; вывод **1440p**; обязательная планка — **стабильные 70–80 fps** в типовых сценах; 144 fps — верхняя цель, а не критерий приёмки.
- Внутренний рендер: **720p–900p** + TAAU/FSR; native 1440p — только для 60 fps.
- Тяжёлое разрушение на арене: не ниже **70–80 fps**, **60 floor**; выше — сколько получится.
- VRAM: **≤6.5 GB из 8 GB**; воксели ≤2.5 GB; render targets ≤1.8 GB; GI ≤0.5 GB; вся арена resident, без streaming.
- CPU: ориентир 6 ядер/12 потоков, 16 GB RAM (у пользователя Ryzen 5 5600G, 32 GB); бюджеты — в `docs/ENGINE_PLAN.md`, раздел 9.
- Платформы: Windows DX12 (fallback Vulkan), Linux Vulkan (NVIDIA + Mesa RADV/ANV).
- Стиль: **style-agnostic**; база PBR + stylized-флаги. Не фиксировать один арт-дирекшен в движке.

## 4. Архитектурные инварианты

- Rust 2024, wgpu 30.x, winit, egui, hecs, glam, Jolt 5.6 через `oxijolt` (fallback Rapier), kira, mlua, serde/ron/bincode, zstd.
- Крейты: `shatterbox-core`, `shatterbox-platform`, `shatterbox-gpu`, `shatterbox-voxel`, `shatterbox-render`, `shatterbox-physics`, `shatterbox-world`, `shatterbox-destruction`, `shatterbox-editor`, `shatterbox-audio`, `shatterbox-script`, `shatterbox-game`, `shatterbox-tools`.
- Зависимости: `shatterbox-game` зависит от всех; **никто не зависит от `shatterbox-game`**. `shatterbox-gpu` — единственное место, где живут типы wgpu; геймплей не видит `wgpu::*`.
- Физика — за трейтом `PhysicsBackend`; игра не знает, Jolt внутри или Rapier.
- Воксель — `u16` (10 бит материал + 6 бит вариант цвета; альтернатива 8+8). Материал — запись палитры: albedo, roughness, metalness, emissive, opacity, плотность, твёрдость, горючесть, звук.
- Мир — sparse brick map: кирпич 16³, регион 32×32×32 м, маски занятости 2³/4³/8³/16³, LOD0 12.5 см / LOD1 25 / LOD2 50; арена 8×2×8 регионов.
- Симуляция — фиксированные 60 Гц; рендер интерполируется; `RenderWorld` double-buffered, геймплей не пишет в GPU напрямую.
- Координаты: арена 256 м укладывается в f32; рендер camera-relative. Floating origin не нужен, пока скоуп — арена.
- Правки вокселей — целочисленные детерминированные команды `{tick, seq, region, op, brush, material, seed}`; стабильные ID; фиксированный порядок обхода.
- Платформы: только Windows + Linux. Никаких console-specific путей, Metal, PS5, Switch.
- Стиль: не хардкодить один вид; toon/unlit/flat/outline/LUT — флаги материалов/поста.

## 5. Структура репозитория

```
docs/ENGINE_PLAN.md      — архитектура и дорожная карта
AGENTS.md                — этот файл
README.md                — краткое описание
.agents/skills/          — установленные и проектные скиллы
crates/
  shatterbox-core/       — math, time, ids, config, error types
  shatterbox-platform/   — window, input, OS, packaging
  shatterbox-gpu/        — wgpu device/queue, resources, pipelines, uploads
  shatterbox-voxel/      — palette, brick map, regions, DDA data, LOD
  shatterbox-render/     — passes, G-buffer, lighting, GI, post, TAAU
  shatterbox-physics/    — PhysicsBackend, Jolt adapter, collision sectors
  shatterbox-world/      — ECS world, arena load/save, prefabs
  shatterbox-destruction/— edits, flood fill, components, debris
  shatterbox-editor/     — egui editor, brushes, import/export
  shatterbox-audio/      — kira mixer, spatial audio
  shatterbox-script/     — Lua API, mods
  shatterbox-game/       — game rules, player, bots, tools, weapons, UI
  shatterbox-tools/      — CLI: cook, pack, import, validate
```

Уже есть: `shatterbox-core`, `shatterbox-platform`, `shatterbox-gpu`, `shatterbox-voxel`, `shatterbox-physics`, `shatterbox-destruction`, `shatterbox-render`, `shatterbox-game`. Остальные крейты создаются на своих фазах; пустые заготовки «на будущее» не создавай.

Текущий статус, замеры и откаты: `docs/ENGINE_PLAN.md`, раздел 10 (10.1–10.3). Активны Фаза 4 (свет, осталось DDGI/огонь/дым) и Фаза 5 (игра: оружие, боты, редактор).

## 6. Правила кода

- `cargo fmt` и `cargo clippy --workspace --all-targets -- -D warnings` — обязательны.
- Ошибки: `thiserror` в библиотеках, `anyhow` в бинарниках/CLI; в рантайме не паниковать на пользовательских данных.
- GPU-структуры: `#[repr(C)]` + `bytemuck`; проверять выравнивание против WGSL.
- `unsafe` — только в изолированном адаптере (Jolt/FFI), с комментарием и безопасной обёрткой.
- Hot path без аллокаций: переиспользуемые буферы, `SmallVec`, арены; не `format!`/`Vec::new()` на кадр.
- Логи: `tracing` spans, не `println!` в кадре; статистика — через оверлей/puffin.
- Никакого blocking I/O в кадре: загрузка — worker-пул + staging ring.
- Документировать публичные API и неочевидные инварианты; комментарии — по делу, не пересказ кода.
- Шейдеры — WGSL, hot reload; общие структуры — через `naga_oil`/include, а не копипасту.
- Не тащить `bevy` целиком; `hecs` достаточно. Не тащить `wgpu` в `shatterbox-game`.

## 7. Производительность

- Бюджет кадра: **14.3 мс на 70 fps / 12.5 мс на 80 fps** — обязательная планка; **6.94 мс на 144 fps** — верхний тир (CPU ~2.8 мс, GPU ~5.9 мс, см. план 9.2).
- Порядок деградации frame governor (цель): GI/volumetrics/specular → лучи теней/AO → render scale → LOD bias. С гистерезисом, не каждый кадр. Сейчас реализован только render scale.
- 144 fps: 1 луч visibility, тени/AO half/quarter, specular half/off, GI 1 каскад/1 отскок, volumetrics off.
- Сначала профилируй, потом оптимизируй: GPU timestamps, Tracy/puffin, PIX/RenderDoc.
- Performance gates в каждой фазе; не откладывать оптимизацию на «потом».
- Не предлагать native 1440p/144 как основной режим — апскейл часть архитектуры.

## 8. Тестирование

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p shatterbox-game
cargo run -p shatterbox-editor   # когда появится
```

- Unit-тесты: адресация вокселей/кирпичей/регионов, маски, DDA на CPU-референсе, палитра, command log, flood fill, LOD.
- Determinism-тесты: одинаковые команды → одинаковый хеш мира на двух прогонах.
- GPU smoke: headless-тест создания устройства/пайплайна; визуальные тесты — вручную.
- CI: Windows + Linux, fmt/clippy/test; GPU-тесты не в CI (разные драйверы).
- Для рендера — сравнение кадров «до/после» на фиксированной сцене, а не на глаз.

## 9. Скиллы

Установлены в `.agents/skills/` и доступны через инструмент `skill`:

- `gamedev-wgpu-rendering` — пайплайны, bind groups, текстуры, surface, устройство.
- `gamedev-wgsl-shaders` — WGSL, layout, uniform alignment, hot reload.
- `gamedev-profiling` — puffin/tracing, GPU timings, поиск бутылочных горлышек.
- `gamedev-rust-workspace` — Cargo workspace, CI, clippy, error strategy.
- `gamedev-winit-loop` — event loop, fixed timestep, resize, input.
- `gamedev-debug-ui` — egui-оверлеи, инспекторы, переключатели, puffin UI.

Проектные скиллы (созданы в `.agents/skills/`):

- `voxel-world` — формат вокселей, brick map, регионы, LOD, детерминизм.
- `voxel-render` — проходы, G-buffer, DDA-шейдер, DDGI, TAAU, governor.
- `voxel-physics` — Jolt/oxijolt, гибридная коллизия, разрушение.
- `voxel-vfx` — взрывы, частицы, копоть, тряска, снаряды и вьюмодели, артефакты света.
- `voxel-autotest` — запуск игры агентом: сценарии, скриншоты, report.json, логи, A/B.

`voxel-open-world` вынесен в `docs/future/` (вне скоупа, см. §11); не загружай его.

Правила: сначала загрузи релевантный скилл, потом делай. Если скилл противоречит плану — план главнее, скилл обнови.

## 10. Workflow агента

1. Прочитай `AGENTS.md` и нужный раздел `docs/ENGINE_PLAN.md`.
2. Загрузи 1–2 релевантных скилла (не все подряд).
3. Составь короткий todo (3–7 пунктов), если задача многошаговая.
4. Сделай минимальный вертикальный срез; не расширяй скоуп.
5. Запусти fmt/clippy/test; для рендера/физики — профиль и замер.
6. Обнови план/доки, если изменилась архитектура, формат или бюджет.
7. Отчитайся: что сделано, какие файлы, какие тесты, какие риски/вопросы.

## 11. Non-goals (сейчас)

- Консоли, Metal, мобилки.
- Открытый мир, streaming, LOD-регионы за пределами арены, floating origin.
- Rollback-неткод, матчмейкинг, античит, выделенные серверы.
- Полноценный GI на Radiance Cascades в первой версии (только R&D).
- Фотореализм как обязательный стиль.
- Скелетная анимация/сложные меши в v0.1 (воксельные жёсткие части, как в Teardown).

## 12. Открытые вопросы

См. `docs/ENGINE_PLAN.md`, раздел 12: формат матча (5 ботов vs 5×5), 60 vs 100+ в тяжёлом разрушении, первый арт-стиль среза, Linux-матрица и темп, первое оружие/инструменты. Не принимай за пользователя решения, которые меняют эти параметры.
