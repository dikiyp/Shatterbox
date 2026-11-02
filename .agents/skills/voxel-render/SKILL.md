---
name: voxel-render
description: Use when working on Shatterbox rendering: wgpu compute passes, voxel DDA visibility, G-buffer, deferred PBR lighting, shadows, AO, GI (SSGI/VCT/probe volume), SSR, fog, FSR1/TAA upscaling, frame governor, or render performance and A/B measurements. Trigger on render pass, shader, gbuffer.wgsl, lighting.wgsl, G-buffer, GI, FSR, resolution scale, GPU timing.
---

# voxel-render

Рендер без треугольников: один DDA-луч на пиксель по brick map, deferred-освещение, half-res эффекты, FSR1-апскейл.

## Приоритет источников

1. Код в `crates/shatterbox-render` и `crates/shatterbox-voxel/src/gpu.rs` — фактическое состояние.
2. `docs/ENGINE_PLAN.md` разделы **10 и 10.1–10.3** — что реально сделано, замеры, откаты.
3. `docs/ENGINE_PLAN.md` раздел 6 — **целевой дизайн, частично устарел** (см. «Расхождения»). Не реализуй по разделу 6 то, что в разделе 10 уже заменено.
4. `AGENTS.md` — бюджеты и инварианты (цифры бюджетов только там).

Скиллы `gamedev-wgpu-rendering` и `gamedev-wgsl-shaders` — для API/шейдерных мелочей; версии крейтов в их примерах устарели, брать из корневого `Cargo.toml`.

## Состояние на конец раздела 10.3 (проверь по коду, если сомневаешься)

Шейдеры и роль:

| Файл | Что делает |
|---|---|
| `gbuffer.wgsl` | primary DDA, **один луч на пиксель**; пишет палитровый цвет с тинтом, нормаль, глубину, `material.b` = индекс материала; контактный AO `1/(1+d²·0.30)` + per-voxel AO только где различим; `cast_debris` по общему AABB обломков |
| `shadow.wgsl` | half-res тени от солнца (2×2 пикселя на луч), дизер без зависимости от номера кадра, фейд в 1.0 на 0.70–0.98 от `shadow_distance` |
| `light_shadow.wgsl` | half-res тени локальных источников, R32Uint, бит на источник |
| `probe.wgsl` | irradiance-объём 32×8×32 (ячейка 8 м) по occupancy кирпичей; альфа = `sky_vis`; пересчёт только при `probe_dirty` |
| `gi.wgsl` | один half-res проход: SSAO + SSGI + упрощённый VCT (3 конуса × 3 тапа) + SSR; rgb = indirect, a = AO; отдельный `reflection` HDR |
| `fog.wgsl` | half-res volumetric fog, 14 шагов, height fog, HG g=0.62; rgb = inscatter, a = transmittance |
| `lighting.wgsl` | deferred PBR (GGX/Cook-Torrance), до 12 point lights + capsule/rect area, процедурная стилизация (`material_detail`, `macro_variation`, `micro_jitter`), композит fog, стекло, небо |
| `fsr.wgsl` | FSR1 EASU+RCAS, слот 9, апскейлер по умолчанию (`SHATTERBOX_UPSCALER=taa` или `U` — TAAU) |
| `post.wgsl` | bloom, ACES, grade, vignette, dither |

Решения, которые уже приняты (не откатывай без замера):

- Sub-pixel jitter сейчас `[0, 0]`, а история в temporal-проходе весит ~89 % (`HISTORY_BLEND = 0.11`) — по аудиту это неработающая конфигурация (размытие и гостинг, стохастика не усредняется). Возврат джиттера допустим **только** вместе с rejection по глубине/нормали и reactive mask (эти правки — одна задача Э1) и только с A/B на p50. Апскейл по умолчанию — FSR1.
- DDGI в основном пути **нет**: вместо него SSGI + VCT по probe-объёму. DDGI 2 каскада — открытая задача Фазы 4.
- Второго (теневого) луча в G-buffer нет: тени только в `shadow.wgsl`/`light_shadow.wgsl`.
- Стилизация живёт в `lighting.wgsl`, не в G-buffer: primary-ядро должно оставаться тонким (регистровое давление).
- Выключенный эффект пропускает dispatch и пишет нейтральное значение; bind groups не меняются.
- Эффекты переключаются в оверлее F1 и `SHATTERBOX_EFFECTS` (биты Фазы 3: 1 тени, 2 SSAO, 4 SSGI, 8 VCT, 16 TAA, 32 fog; после 10.1 нумерация могла измениться — сверь по коду).

## Расхождения: план §6 против реализации

- §6.1/6.9/6.10: «TAAU, джиттер, motion vectors» → факт: FSR1, без джиттера, depth-reprojection без motion vectors.
- §6.3: G-buffer из 5 таргетов (RT0–RT4) → факт: 4 таргета (albedo+AO, normal+roughness, metallic/emissive/material id, ray depth); после 10.2 состав мог измениться. **Точная раскладка — только из кода.**
- §6.5: DDGI как основа GI → факт: SSGI + VCT + probe-объём.
- §6.6: specular через DDA-луч → факт: SSR в `gi.wgsl` (18 шагов), промах → аналитическое небо.
- Governor в плане «меняет scale, лучи, частоту GI» → в разделе 10 описан только **render scale**. Порядок деградации из `AGENTS.md` (GI → тени/AO → scale → LOD) пока не реализован целиком.

Если правишь рендер так, что это расхождение меняется, обнови раздел 6 плана (правило проекта: сначала план, потом код).

## Frame governor (как реализован)

- EMA игнорирует кадры длиннее `target * 2.5` (перезагрузка шейдеров, композитор).
- Выдержка 1.5 с, мёртвая зона: вниз при `> +12 %` от бюджета, вверх при `< −10 %`; шаг 0.05; пол 0.35; старт 0.65.
- `ensure_targets` кэширует последний набор `Targets`/`BindGroups`; переключение 0.75 ↔ 0.80 не должно переаллоцировать сотни МБ.
- `history_reset` при любом изменении internal или output size, при переключении TAA и смене jitter-паттерна.
- Пресеты `SHATTERBOX_QUALITY=performance|balanced|quality`: дальность теней 56/112/160 м.

## DDA

- Иерархия: регион 32 м → группа 8 м → кирпич 2 м → маски 1 м / 0.5 м / 0.25 м → mask16 → воксель. Без делений в горячем цикле.
- `t_exit` из `intersect_world` и жёсткий `break` на выходе из мира: лучи в небо не должны крутить `MAX_STEPS`.
- `MAX_STEPS = 512` (максимум 205 шагов на реальных лучах; тест `grazing_rays_fit_the_shader_step_budget`).
- Primary ray до 192 м (128 м давал видимую сферу обрезания — не уменьшать).
- Любая правка DDA — сначала в CPU-реплике `shatterbox-voxel/src/gpu.rs` и тесте, потом в WGSL.

## Известные ловушки (все уже случались в этом проекте)

- `block_boundary` считал границу только для луча «вперёд» и для нисходящего primary-луча прыгал за цель → shading point внутри плиты, «прозрачные полосы», крапинка теней.
- Обновление `t_max` после coarse-прыжка: нужен `|cell - previous|`, не знаковая дельта; иначе на обратных осях время пересечения уходит в прошлое.
- `Brick::set_mask_bit` при удалении вокселя сбрасывал биты целого coarse-блока → лучи проваливались под пол. Mask8/4/2 при снятии бита пересчитываются как OR дочерних уровней.
- Обломки: локальная DDA-сетка обязана учитывать отрицательный `local_min` (сдвиг на `aabb_min`, кламп ячеек в `0..dims`).
- Выстрел и рендер должны брать одну и ту же `aim_camera` (screen shake — только rotational); тест `center_pixel_matches_camera_forward`.
- История TAA/FSR при смене размера буферов без `history_reset` → «тряска».
- Пространственный хеш по мировой позиции для дизера теней → периодические волны; использовать blue-noise пикселя.
- SSGI читает HDR **предыдущего** кадра (иначе обратная связь с lighting).
- Шейдер чувствителен к регистрам: даже безобидный early-out/кэш в `gbuffer.wgsl` может стоить +0.3 мс.

## Известные проблемы по аудиту (подтверждай по коду перед правкой)

- `FrameUniform` (Rust, `renderer.rs`) и WGSL (`lighting.wgsl`, `light_shadow.wgsl`) расходятся по порядку полей: `light_axis` в шейдере стоит сразу после `light_color`, в Rust — после `debris_max`. Следствие: capsule/rect вырождаются в point. Лечить порядком полей и явными `min_binding_size` вместо `None`.
- AO, GI, fog, reflection читаются в `lighting.wgsl` как `pixel/2` nearest из half-res → блочность. Нужен depth/normal-aware апсэмплинг.
- SSR включается битом SSGI; SSGI и SSR читают историю, в которую пишут друг друга (самоподкормка, «светящиеся» углы); SSGI без репроекции истории.
- Отражения читаются в `lighting`, даже если GI-проход пропущен → «замороженные» отражения.
- `probe_dirty` не выставляется при ресайзе/смене scale → VCT и лучи в тумане чёрные до правки мира.
- AO в G-буфере обходит 4 соседей и обрывается на границе кирпича → шов каждые 2 м.
- Индекс материала 10 бит в вокселе, 8 бит в G-буфере; `opacity` не попадает в G-буфер (стекло шейдится как непрозрачное).
- Нет timestamp'ов на `shadow`, `light_shadow`, `fog`, `gi`, bloom; без них нельзя понять, какой проход съел бюджет.
- Возможна двойная гамма, если выбран sRGB-формат свопчейна (`context.rs`) при ручной гамме в `post.wgsl`.

## Что уже пробовали и откатили

- Глобальная пирамида занятости 4 м над кирпичами: 4.83–4.91 → 5.23–5.40 мс (хуже). Хвост луча тратится внутри непустых кирпичей, где уровень 1 м уже есть.
- Early-out в `cast_debris` + кэш mask2: +0.3 мс.
- Не повторяй без нового аргумента и A/B.

## Замеры (обязательный порядок)

- Конфигурация: RTX 3050, DX12, release, 2560×1440 output, `SHATTERBOX_NO_VSYNC=1`, только `Presented`-кадры.
- Сравнивать **p50** в одинаковых условиях; шум на машине до ±30 % (фоновые приложения, GPU-клиенты). Перед A/B закрыть лишнее, гонять 3–4 с повторно.
- Смотреть GPU ms по проходам (gbuffer / lighting / temporal), а не только fps.
- A/B: одна сцена, один набор эффектов, один scale; фиксировать «до/после» в таблицу раздела 10.x плана.
- Опорные цифры (раздел 10.3, native 1440p, всё включено): 80.9 fps, GPU 12.27 мс (gbuffer 6.26, lighting 0.90, temporal 0.73); с `SHATTERBOX_EFFECTS=0` — 116.3 fps. Gate 144 fps = 6.94 мс, достигается только через internal scale + FSR.
- Бенчмарк пока не покрывает активное разрушение; не заявляй «100+ fps в разрушении» без ручного/скриптового замера.

## Открыто (Фаза 4)

- DDGI 2 каскада вместо SSGI/VCT, огонь/дым, WBOIT-стекло на несколько слоёв.
- Узкое место native 1440p — primary DDA в `gbuffer`; идеи не из откаченного: half-res проход видимости или воксельная shadow map (давали артефакты на тонких укрытиях).

## Перед завершением задачи

1. `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
2. Для шейдеров: нет ошибок валидации wgpu; выравнивание структур Rust ↔ WGSL.
3. Замер до/после (см. выше) и запись в `docs/ENGINE_PLAN.md`, раздел 10.x.
