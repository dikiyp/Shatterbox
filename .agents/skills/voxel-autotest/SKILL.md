---
name: voxel-autotest
description: Use when the agent must run Shatterbox itself: launching the game safely with a timeout, running scripted scenarios, capturing screenshots, reading report.json and logs, detecting wgpu validation errors, and doing A/B comparisons of render changes. Trigger on run the game, screenshot, scenario, report.json, validation error, benchmark, A/B, regression.
---

# voxel-autotest

Агент сам запускает игру, проверяет логи, делает скриншоты и сравнивает до/после. Пользователь не тестирует руками.

## Базовые правила запуска

1. **Всегда с таймаутом и закрытием процесса.** Никогда не запускай `cargo run` без ограничения времени. Шаблон PowerShell:
   ```powershell
   $p = Start-Process -FilePath "target\release\shatterbox.exe" -ArgumentList "--scenario","fire_wall","--out","target\autotest\run1" -PassThru -NoNewWindow -RedirectStandardOutput "target\autotest\run1.out.txt" -RedirectStandardError "target\autotest\run1.err.txt"
   if (-not $p.WaitForExit(120000)) { Stop-Process -Id $p.Id -Force; "TIMEOUT" }
   $p.ExitCode
   ```
2. Один экземпляр игры за раз. Не оставляй процессы после себя (`Get-Process shatterbox` в конце).
3. Сначала `cargo build --release -p shatterbox-game` отдельным шагом; ошибки сборки читай полностью.
4. Режим окна — обычный, не полноэкранный; не трогай чужие окна и не делай OS-скриншотов: только встроенный захват движка.
5. Цифры — только из `report.json` и логов. Не оценивай fps по памяти и не «ожидай» значений.

## Что делает харнесс

`shatterbox.exe --scenario <имя> --out <папка> [--scale 0.7] [--effects N] [--seed S] [--frames N]`

- Сценарии лежат в `scenarios/*.ron`. Шаги: `camera{pos,yaw,pitch}`, `time_of_day`, `effects`, `wait{frames}`, `fire{weapon}`, `screenshot{name}`, `assert{...}`, `exit`.
- Режим фиксированного времени: sim 60 Гц, одинаковый seed, синтетический ввод. Скриншоты воспроизводимы.
- Скриншот — рендер в offscreen-текстуру выходного разрешения, readback, PNG. Не из свопчейна.
- Выходные файлы: `*.png`, `report.json`, `log.txt`.
- Код выхода ≠ 0, если: ошибка валидации wgpu, panic, таймаут, проваленный `assert`.

## report.json (что должно быть)

- `frames`, `fps_p50`, `fps_p99`, `frame_ms_p50`;
- `gpu_ms` по проходам: p50 и p99 (`gbuffer`, `shadow`, `light_shadow`, `gi`, `fog`, `lighting`, `bloom`, `post/fsr`);
- `validation_errors` (число и первые 5 текстов), `warn_count`, `error_count`;
- физика: `bodies`, `awake`, `debris_drawn`, `debris_dropped`;
- мир: `voxels_removed`, `world_hash`, `dirty_bricks`;
- параметры запуска (scale, effects, seed, commit/хеш билда, размеры).

## Как проверять результат

1. Код выхода 0 и `validation_errors == 0`. Любая ошибка валидации — стоп и починка.
2. Посмотри PNG (если ты умеешь читать изображения): опиши, что видно на каждом ключевом кадре, и сравни с критерием этапа. Если не умеешь, скажи об этом в отчёте и опирайся на числа и `diff`; внешний вид в таком случае остаётся на пользователе.
3. Сравнение до/после: `shatterbox-tools diff a.png b.png` → среднее абсолютное отклонение, процент изменённых пикселей, diff-картинка. Для A/B без изменения картинки (оптимизация) отклонение должно быть около нуля.
4. Замеры: 3 прогона подряд, взять медиану p50; шум на машине до ±30 % (фоновые приложения). Сравнивай A/B в одной сессии, чередуя запуски (A B A B). Различия меньше шума не называй улучшением.

## Автоматические проверки, которые стоит держать в сценариях

- `assert_hit_matches_render`: мировая позиция из depth-пикселя под перекрестием совпадает (в пределах 1 вокселя) с CPU-рэйкастом из `aim_camera`.
- `assert_voxel_conservation`: `sum(voxel_count) == removed`.
- `assert_no_floating`: нет вокселей без связи с землёй в стартовом состоянии сцены (когда будет flood fill).
- `assert_debris_visible`: `debris_dropped == 0` при бюджете.
- `assert_hash`: хеш мира после сценария совпадает с эталоном.

## Типовые сценарии

- `smoke`: загрузка, 60 кадров, 1 скриншот, выход. Должен проходить всегда.
- `fire_wall`: стена, 1 ракета, скриншоты до/после и через 1 с.
- `fire_series`: 5 ракет с интервалом 0.25 с, бенчмарк разрушения (p50/p99, число тел).
- `lighting_set`: 4 позиции камеры, фиксированные день/ночь, для теней и AO.
- `edge_set`: границы объект/небо и объект/пол, для лесенки/halo.

## Типичные ошибки

- Запуск без таймаута → зависший процесс блокирует GPU.
- Сравнение скриншотов с разным seed/временем → ложные диффы.
- Вывод fps из оверлея вместо `report.json`.
- «Всё хорошо», когда в логе есть `ERROR`, но игра не упала (пайплайн мог откатиться на встроенные шейдеры): считай `error_count`.
