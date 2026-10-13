#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_frames_by_DI_GI.py
=========================

Скрипт для отбора кадров молекулярно-динамической траектории по значениям
двух коллективных переменных:

    DI (Dimer Index)  = sum_{i<j} [ -cos(theta_ij) / (1 + d_ij^12) ]
    GI (Growth Index) = sum_{i<j} [  sin(theta_ij) / (1 + d_ij^12) ]

где суммирование идёт по всем 15 парам i < j шести мономеров калексарена
(пирогаллоарен или резорцинарен), d_ij — расстояние между геометрическими
центрами «центральных кислородов» мономеров i и j (с учётом минимального
изображения), theta_ij — угол между осями мономеров a_i = C_i - O_i
(центры tailC минус центральные O).

ВАЖНО ПРО ЕДИНИЦЫ В ФОРМУЛЕ ВЕСА 1/(1 + d^12):
  * Как и в эталонной постановке PLUMED/GROMACS, d_ij в этой формуле
    подставляется в НАНОМЕТРАХ (при d = 1 нм вес равен ровно 1/2).
  * Единицы входных файлов НЕ гарантированы: данная траектория
    (superlight_traj.xtc / ref_nosolv.gro) записана в АНГСТРЕМАХ —
    бокс ~598x964x708 Å при мономере размером ~4 нм. MDAnalysis отдаёт
    эти числа «как есть», и если подставить их в формулу напрямую,
    d^12 ~ 1e15..1e18, вес схлопнется в ~0, и на всех кадрах получатся
    DI ~ 0.004 и GI ~ 0.0002 вместо ожидаемых величин порядка единиц.
    Именно эта ошибка (смешение ангстремных координат с нанометровой
    формулой веса) была причиной «нулевых» значений.
  * Исправление: скрипт АВТОМАТИЧЕСКИ определяет единицы по физическому
    размеру мономера и, если координаты пришли в ангстремах, переводит
    их (и бокс) в нанометры перед вычислом центров, осей и d_ij. После
    этого вся геометрия считается в нм, а формула 1/(1 + d^12) получает
    расстояние именно в нанометрах.
  * Углы theta_ij от единиц не зависят вообще.

Важные детали алгоритма (полностью повторяют постановку PLUMED):
  * Мономеры перед вычислением центров «делаются целыми» (аналог
    WHOLEMOLECULES), иначе мономер, пересекающий периодическую границу,
    даст неверный центр.
  * Используются ГЕОМЕТРИЧЕСКИЕ центры (CENTER без MASS), а не центры масс.
  * Координаты и бокс из .xtc/.gro заданы в нм; PBC-вычисления ведутся в нм.
  * cos_theta ограничивается диапазоном [-1, 1] перед вычислением sin.

Выход скрипта:
  * CSV-файл со всеми кадрами: frame, time_ps, DI, GI;
  * XTC-файл с отобранными кадрами (в исходном порядке, все атомы системы);
  * Timeline — полоса времени всей траектории (серая), на которой ярким
    цветом выделены интервалы кадров, попавших в selected (см. ниже);
  * (опционально) 2D-гистограмма (DI, GI) с целевой точкой и окном допуска;
    цветовая шкала тепловой карты — РАЗНОЦВЕТНАЯ ДИСКРЕТНАЯ («радуга»):
    много разных цветов, идущих друг за другом (по умолчанию до ~64 уровня,
    каждый цвет = небольшой интервал популяции), упорядоченных интуитивно —
    от холодного фиолетового (мало кадров) через синий/зелёный/жёлтый/оранжевый
    к красному (максимум); число цветов настраивается --heatmap_levels;
  * краткий отчёт в stdout.

TIMELINE (полоса времени):
  * Серая горизонтальная полоса = всё время исходной траектории
    (от первого до последнего кадра).
  * Яркие цветные полосы поверх неё = промежутки времени, в которые система
    находилась в нужной области значений DI/GI (кадры из selected соседние
    кадры склеиваются в непрерывные интервалы).
  * Строится всегда при полном расчёте (--out_timeline_html / --out_timeline_png;
    по умолчанию — рядом с --out_csv). Если plotly не установлен,
    интерактивный HTML пропускается с предупреждением; статический PNG
    строится matplotlib'ом.
  * Интерактивная версия (HTML, plotly): при наведении курсора на яркий
    интервал во всплывающей подсказке показывается его начало/конец
    (время и номера кадров), длительность и число отобранных кадров;
    при наведении на серую полосу — общая длительность траектории.

Пример запуска:

    python3 select_frames_by_DI_GI.py \
        --gro ref_nosolv.gro \
        --xtc traj.xtc \
        --ndx_O index_singoli_O.ndx \
        --ndx_C index_singoli_C.ndx \
        --DI -0.5 --GI 0.3 \
        --tol_DI 0.2 --tol_GI 0.3 \
        --out_xtc selected.xtc \
        --out_csv all_frames.csv \
        --plot --out_png fes_hist.png

    # Вместо жёсткого окна — выбрать 10 ближайших кадров к целевой точке:
    python3 select_frames_by_DI_GI.py ... --top_n 10

    # Перестроить только heatmap по уже рассчитанному CSV — БЕЗ пересчёта траектории:
    python3 select_frames_by_DI_GI.py --from_csv all_frames.csv \
        --DI 0.7 --GI 11.0 --out_png fes_hist.png

Зависимости: numpy, MDAnalysis (matplotlib — только для опций --plot / --from_csv).
"""

import argparse
import math
import os
import re
import sys

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.lib import distances as mdadist
except ImportError:  # pragma: no cover
    sys.stderr.write("Ошибка: не найден пакет MDAnalysis. Установите его: pip install MDAnalysis\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Константы задачи
# ---------------------------------------------------------------------------
N_MONOMERS = 6          # шесть одинаковых мономеров
ATOMS_PER_MONOMER = 76  # каждый мономер занимает 76 атомов подряд

# Единицы: вся геометрия должна считаться в НАНОМЕТРАХ — именно так устроен
# эталонный PLUMED-расчёт (GROMACS передаёт PLUMED координаты в нм), и только
# в нм формула 1/(1 + d^12) физически осмысленна (при d = 1 нм вес = 1/2;
# при d ~ 15..30 Å d^12 ~ 1e15..1e18, вес схлопывается в ноль, и DI/GI везде
# выглядят как 0.000).
#
# ОДНАКО входные файлы могут быть записаны НЕ в нм. Данная траектория
# (superlight_traj.xtc / ref_nosolv.gro) записана в АНГСТРЕМАХ: бокс
# ~598x964x708 при мономере каликсарена размером ~4 нм. Поэтому перед
# расчётом скрипт автоматически определяет единицы по физическому размеру
# мономера и при необходимости переводит координаты и бокс из Å в нм
# (см. detect_length_unit / ANGSTROM_TO_NM).
ANGSTROM_TO_NM = 0.1

# Физический «размер» одного мономера каликс[4]арена (макс. внутримолекулярное
# расстояние): ~3.5–4.5 нм. Если медианный размер мономера в координатах
# кадра близок к ~40, значит координаты заданы в ангстремах.
MONOMER_SIZE_NM_TYPICAL = 4.0     # типичный диаметр мономера, нм
UNIT_DETECT_THRESHOLD   = 15.0    # порог: медианный размер > 15 => это ангстремы


# ---------------------------------------------------------------------------
# Чтение GROMACS .ndx файла
# ---------------------------------------------------------------------------
def read_ndx(path):
    """Разбирает index-файл GROMACS вручную и возвращает dict {имя группы: [индексы]}.

    Индексы возвращаются как есть — 1-based, как записаны в файле.
    Преобразование к 0-based выполняется на стороне вызывающего кода.

    Обрабатываются крайние случаи:
      * файла не существует -> FileNotFoundError;
      * пустой файл / файл без групп -> ValueError.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Index-файл не найден: {path}")

    groups = {}
    current_name = None       # имя текущей группы
    current_idx = []          # накопитель индексов текущей группы

    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\[\s*(.+?)\s*\]$", line)  # строка вида "[ cal1_centralO ]"
            if m:
                # закрыли предыдущую группу
                if current_name is not None:
                    groups[current_name] = current_idx
                current_name = m.group(1)
                current_idx = []
            else:
                # строка с числами (индексы могут быть разделены пробелами/табами)
                if current_name is None:
                    continue  # числа до первой группы — игнорируем
                nums = line.split()
                for n in nums:
                    try:
                        current_idx.append(int(n))
                    except ValueError:
                        # мусорная строка — пропускаем
                        pass

    if current_name is not None:
        groups[current_name] = current_idx

    if not groups:
        raise ValueError(f"Index-файл пуст или не содержит ни одной группы: {path}")

    return groups


def get_group_indices(groups, name, ndx_path, natoms):
    """Достаёт группу `name` из словаря групп, проверяет её и переводит в 0-based."""
    if name not in groups:
        raise KeyError(
            f"В index-файле '{ndx_path}' отсутствует группа '{name}'. "
            f"Доступные группы: {sorted(groups.keys())}"
        )
    idx = np.asarray(groups[name], dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"Группа '{name}' в файле '{ndx_path}' пуста.")
    if idx.min() < 1 or idx.max() > natoms:
        raise ValueError(
            f"Группа '{name}' содержит индексы вне диапазона 1..{natoms} "
            f"(min={idx.min()}, max={idx.max()}) — число атомов в топологии "
            f"не совпадает с index-файлом."
        )
    return idx - 1  # преобразование 1-based (.ndx) -> 0-based (MDAnalysis)


# ---------------------------------------------------------------------------
# Автоопределение единиц входной траектории (нм или ангстремы)
# ---------------------------------------------------------------------------
def detect_length_unit(coords, box_lengths, first_atoms):
    """Определяет, в каких единицах заданы координаты кадра: 'nm' или 'angstrom'.

    Критерий — физический размер одного мономера (максимальное внутримолекулярное
    расстояние). Мономер каликс[4]арена имеет диаметр ~3.5–4.5 нм; если в файле
    его «размер» получился ~35–45, значит координаты записаны в ангстремах.

    Параметры:
        coords      : (N, 3) координаты кадра «как есть» из файла
        box_lengths : (3,) длины векторов бокса «как есть» из файла
        first_atoms : 0-based индексы первых атомов мономеров
    Возвращает:
        ('angstrom', monomer_size) или ('nm', monomer_size)
    """
    sizes = []
    for start in first_atoms:
        seg = coords[start:start + ATOMS_PER_MONOMER]
        if len(seg) >= 2:
            # грубая оценка диаметра: диагональ ограничивающего прямоугольника
            span = seg.max(axis=0) - seg.min(axis=0)
            sizes.append(float(np.linalg.norm(span)))
    med_size = float(np.median(sizes)) if sizes else 0.0
    # дополнительный ориентир: размер бокса (для этой системы ~60x96x71 нм
    # или ~598x964x708 Å)
    if med_size > UNIT_DETECT_THRESHOLD or float(np.median(box_lengths)) > 10 * MONOMER_SIZE_NM_TYPICAL * 5:
        return "angstrom", med_size
    return "nm", med_size


# ---------------------------------------------------------------------------
# Шаг 1. «Сделать молекулы целыми» (аналог WHOLEMOLECULES из PLUMED)
# ---------------------------------------------------------------------------
def make_molecules_whole(coords_nm, box, first_atoms):
    """Раскладывает каждую молекулу целиком внутри периодической ячейки.

    Алгоритм: идём по атомам молекулы последовательно; если соседний атом
    оказался дальше чем на половину бокса от предыдущего (т.е. пересёк
    границу), сдвигаем его и всех последующих атомы молекулы на вектор
    решётки. Для ортогонального бокса это обычный minimum-image сдвиг;
    для триклинического — разложение смещения по базису решётки
    (triclinic_vectors / transform_RtoS в MDAnalysis).

    Параметры:
        coords_nm   : (N, 3) координаты в нм (будет скопирован, исходное не меняется)
        box         : (3, 3) матрица.box'а кадра из ts.dimensions
        first_atoms : список 0-based индексов первых атомов каждой молекулы
    Возвращает:
        (N, 3) «починенные» координаты в нм.
    """
    c = coords_nm.copy()
    triclinic = np.abs(box[0, 1]) + np.abs(box[0, 2]) + np.abs(box[1, 2]) > 1e-6

    for start in first_atoms:
        stop = min(start + ATOMS_PER_MONOMER, len(c))
        seg = c[start:stop]
        if len(seg) < 2:
            continue
        # вектора связей между последовательными атомами молекулы
        bonds = np.diff(seg, axis=0)
        if triclinic:
            # переводим декартовы смещения в фракционные координаты решётки
            frac = mdadist.triclinic_vectors(box, bonds)
            # компонентa, вылезшая за [-0.5, 0.5], указывает на пересечение границы
            wrap = np.round(frac)
        else:
            lengths = np.diag(box)  # a, b, c
            wrap = np.zeros_like(bonds)
            for k in range(3):
                wrap[:, k] = np.round(bonds[:, k] / lengths[k])
        if not np.any(wrap):
            continue  # молекула уже цела
        # накапливаем сдвиги: атом m получает сумму поправок всех связей до него
        cumshift_frac = np.vstack([np.zeros((1, 3)), np.cumsum(wrap, axis=0)])
        if triclinic:
            shift = mdadist.transform_StoR(box, -cumshift_frac)
        else:
            shift = -cumshift_frac * np.diag(box)[None, :]
        c[start:stop] += shift
    return c


# ---------------------------------------------------------------------------
# Шаги 2–6. Вычисление DI и GI для одного кадра
# ---------------------------------------------------------------------------
def compute_DI_GI(coords_nm, box_dims_nm, o_idx, c_idx):
    """Вычисляет DI и GI для одного кадра. ВСЕ ЕДИНИЦЫ — НАНОМЕТРЫ.

    Параметры:
        coords_nm     : (456, 3) координаты атомов системы в нм
                        (мономеры уже «сделаны целыми»);
        box_dims_nm   : размеры бокса кадра в нм в формате ts.dimensions —
                        [lx, ly, lz, alpha, beta, gamma]; именно его принимают
                        функции MDAnalysis.lib.distances;
        o_idx         : список из 6 массивов 0-based индексов centralO каждого мономера;
        c_idx         : список из 6 массивов 0-based индексов tailC каждого мономера.
    Возвращает:
        (DI, GI) — два float.
    """
    # --- Шаг 2: геометрические центры (НЕ центры масс!) ---------------------
    O = np.array([coords_nm[g].mean(axis=0) for g in o_idx])   # (6, 3), нм
    C = np.array([coords_nm[g].mean(axis=0) for g in c_idx])   # (6, 3), нм

    # --- Шаг 3: оси мономеров ----------------------------------------------
    a = C - O                                                   # (6, 3), нм
    norm_a = np.linalg.norm(a, axis=1)                          # (6,)

    # --- Шаг 4: попарные расстояния d_ij между центрами O_i -----------------
    # self_distance_array даёт все пары i<j с учётом minimum image convention
    # относительно бокса текущего кадра. Результат — в нм (тех же, что вход).
    pair_d = mdadist.self_distance_array(O, box=box_dims_nm)       # (15,) пар i<j в нм

    DI = 0.0
    GI = 0.0
    # --- Шаг 4–6: цикл по 15 парам i < j ------------------------------------
    p = 0
    for i in range(N_MONOMERS):
        for j in range(i + 1, N_MONOMERS):
            d = float(pair_d[p]); p += 1      # расстояние в НМ

            # скалярное произведение осей учёт PBC НЕ требует (это векторы-направления)
            dot = float(np.dot(a[i], a[j]))
            denom = norm_a[i] * norm_a[j]
            if denom < 1e-12:
                raise ZeroDivisionError(
                    f"Нулевая длина оси мономера (пара {i+1},{j+1}) — проверьте index-группы."
                )
            cos_t = dot / denom
            cos_t = max(-1.0, min(1.0, cos_t))          # клиппинг в [-1, 1]
            sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))

            # ВНИМАНИЕ, ЕДИНИЦЫ! d подставляется в формулу 1/(1 + d^12) В НАНОМЕТРАХ —
            # ровно так же, как в эталонной постановке PLUMED/GROMACS (GROMACS передаёт
            # PLUMED координаты в нм). При d = 1 нм вес равен 1/2, при d ~ 1.5..3 нм —
            # доли/сотые, т.е. вклады пар имеют физический масштаб. Если бы сюда
            # попало расстояние в ангстремах (d ~ 15..30 Å), то d**12 ~ 1e15..1e18,
            # вес схлопнулся бы в ~0 и все DI/GI выглядели как 0.000 — это была ошибка
            # ранней версии скрипта. Теперь координаты и бокс КАЖДОГО кадра заранее
            # приводятся к нанометрам в analyze_trajectory (автоопределение единиц),
            # поэтому d здесь гарантированно в нм.
            weight = 1.0 / (1.0 + d ** 12)              # вес пары по расстоянию (d в нм)
            DI += -weight * cos_t                       # cv1_ij
            GI += weight * sin_t                        # cv2_ij

    return DI, GI


# ---------------------------------------------------------------------------
# Основной проход по траектории
# ---------------------------------------------------------------------------
def box_matrix_from_dimensions(dimensions):
    """Возвращает (3,3) матрицу векторов бокса в нм из ts.dimensions.

    dimensions — [a, b, c, alpha, beta, gamma] (углы в градусах).
    Для орторомбического бокса это diag(a,b,c); для триклинического —
    стандартная нижнетреугольная матрица GROMACS/MDAnalysis.
    Возвращает None, если бокс вырожден/отсутствует.
    """
    d = np.asarray(dimensions, dtype=float)
    if d.size != 6:
        return None
    lengths, angles = d[:3], d[3:]
    if np.any(lengths <= 0) or not np.all(np.isfinite(d)):
        return None
    al, be, ga = angles
    if abs(al - 90.0) < 1e-6 and abs(be - 90.0) < 1e-6 and abs(ga - 90.0) < 1e-6:
        return np.diag(lengths)
    from MDAnalysis.lib.util import triclinic_box
    return triclinic_box(*lengths, *angles)


def box_dimensions_from_matrix(box_mat):
    """Из (3,3) матрицы векторов бокса восстанавливает [lx, ly, lz, alpha, beta, gamma].

    Это тот самый формат `box`, который принимают high-level функции
    MDAnalysis.lib.distances (self_distance_array и др.).
    Углы между векторами решётки считаются по скалярным произведениям.
    """
    lx, ly, lz = (np.linalg.norm(box_mat[k]) for k in range(3))
    def ang(u, v):
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    # alpha — между vectors[1] и vectors[2], beta — между [0] и [2], gamma — между [0] и [1]
    alpha = ang(box_mat[1], box_mat[2])
    beta = ang(box_mat[0], box_mat[2])
    gamma = ang(box_mat[0], box_mat[1])
    return np.array([lx, ly, lz, alpha, beta, gamma], dtype=np.float32)


def analyze_trajectory(universe, o_idx, c_idx, progress_every=1000):
    """Возвращает массивы: frames, times_ps, DI, GI (по всем кадрам траектории)."""
    frames, times, dis, gis = [], [], [], []

    first_atoms = [m * ATOMS_PER_MONOMER for m in range(N_MONOMERS)]  # 0-based: 0,76,...,380

    n_total = universe.trajectory.n_frames
    print(f"Всего кадров в траектории: {n_total}, атомов в системе: {universe.atoms.n_atoms}")

    scale = None   # множитель перевода координат/бокса в нанометры
    for ts in universe.trajectory:
        box_raw = box_matrix_from_dimensions(ts.dimensions)   # (3,3) матрица бокса «как в файле»
        if box_raw is None:
            raise ValueError(
                f"Кадр {ts.frame}: в траектории отсутствуют корректные размеры бокса. "
                f"Для расчёта PBC необходим файл с корректной периодичностью (.gro/.tpr)."
            )

        # --- Определение единиц входных данных (по первому кадру) -----------
        # Формула веса 1/(1 + d^12) требует d в НАНОМЕТРАХ. Если траектория
        # записана в ангстремах (наша superlight_traj.xtc: бокс ~598x964x708),
        # то без перевода d^12 ~ 1e15..1e18, вес схлопывается в ноль и все
        # DI/GI получаются ~0.0001 — именно так выглядела исходная ошибка.
        if scale is None:
            unit, med_size = detect_length_unit(ts.positions, np.diag(box_raw), first_atoms)
            scale = ANGSTROM_TO_NM if unit == "angstrom" else 1.0
            if unit == "angstrom":
                print(f"Единицы входной траектории: АНГСТРЕМЫ "
                      f"(размер мономера ~{med_size:.1f}); перевожу координаты и бокс в нм (x{ANGSTROM_TO_NM}).")
            else:
                print(f"Единицы входной траектории: нанометры "
                      f"(размер мономера ~{med_size:.2f} нм); перевод не требуется.")

        # Приводим координаты и бокс кадра к нанометрам.
        pos_nm = ts.positions * scale
        dims_nm = np.asarray(ts.dimensions, dtype=float)
        dims_nm[:3] *= scale
        box_nm = box_raw * scale

        # Шаг 1: чиним молекулы через ПБЦ (в нм).
        whole_nm = make_molecules_whole(pos_nm, box_nm, first_atoms)

        DI, GI = compute_DI_GI(whole_nm, dims_nm, o_idx, c_idx)

        frames.append(ts.frame)
        times.append(ts.time)   # время кадра в пс
        dis.append(DI)
        gis.append(GI)

        # прогресс каждые progress_every кадров
        done = len(frames)
        if done % progress_every == 0 or done == n_total:
            print(f"  обработано {done}/{n_total} кадров ... последний: "
                  f"frame={frames[-1]}, time={times[-1]:.1f} ps, DI={dis[-1]:+.4f}, GI={gis[-1]:+.4f}",
                  flush=True)

    return (np.asarray(frames, dtype=int), np.asarray(times, dtype=float),
            np.asarray(dis, dtype=float), np.asarray(gis, dtype=float))


# ---------------------------------------------------------------------------
# Опциональный график 2D-гистограммы (DI, GI)
# ---------------------------------------------------------------------------
def make_discrete_popularity_cmap(levels, base_name="rainbow"):
    """Строит РАЗНОЦВЕТНУЮ ДИСКРЕТНУЮ цветовую шкалу для тепловой карты.

    Шкала состоит из `levels` РАЗНЫХ цветов, идущих друг за другом (как
    полосы радуги): каждый цвет покрывает свой небольшой интервал значений
    популяции, поэтому ячейки с близкими популяциями окрашиваются заметно
    разными цветами (в отличие от плавного градиента, где они сливаются).

    В качестве базы по умолчанию используется радужная палитра `rainbow`,
    но берётся только её «тёплая» часть (0.125..1.0) — самая первая полоса
    классического rainbow почти неотличима от последней (обе красные), и
    при большом числе цветов соседство таких оттенков путало бы читателя.

    Порядок цветов интуитивно понятен человеку: чем больше кадров в ячейке,
    тем «горячее» её цвет — фиолетовый -> синий -> голубой -> зелёный ->
    жёлтый -> оранжевый -> красный (максимум).

    Возвращает matplotlib ListedColormap с `levels` цветами. Границы уровней
    (bounds) задаёт вызывающий код — см. plot_histogram.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    levels = max(2, int(levels))
    base = plt.get_cmap(base_name)
    if base_name == "rainbow":
        # «тёплая» часть радуги без возврата в красный на обоих концах
        colors = [base(t) for t in np.linspace(0.125, 1.0, levels)]
    else:
        colors = [base(t) for t in np.linspace(0.0, 1.0, levels)]
    cmap = matplotlib.colors.ListedColormap(colors, name=f"pop_{base_name}_{levels}")
    return cmap


def plot_histogram(di_all, gi_all, DI_target, GI_target, tol_DI, tol_GI, out_png,
                   heatmap_levels=None):
    """Строит 2D-гистограмму распределения кадров в плоскости (DI, GI),
    отмечает целевую точку и прямоугольник окна допуска, сохраняет PNG.

    Цветовая шкала — разноцветная дискретная (см. make_discrete_popularity_cmap):
    много разных цветов, идущих друг за другом (радуга), каждый покрывает
    небольшой интервал популяции; порядок интуитивен — холодный фиолетовый
    = редко, горячий красный = часто."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # рендеринг без дисплея
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: matplotlib не установлен, пропускаю построение графика (--plot).")
        return

    # --- 2D-гистограмма (считаем сами, чтобы раскрасить дискретной шкалой) ---
    bins = 60
    H, xe, ye = np.histogram2d(di_all, gi_all, bins=bins)
    max_pop = int(H.max()) if H.size else 1

    if heatmap_levels is None:
        # автовыбор: много разных цветов, каждый покрывает небольшой интервал
        # популяции: при max_pop <= 64 — один цвет на каждое значение
        # (1 кадр/цвет); иначе шаг округляется вверх до «удобного» числа
        # (1, 2, 2.5, 5 x 10^n), а число цветов остаётся большим (десятки)
        raw_step = max(1.0, max_pop / 64.0)
        mag = 10 ** math.floor(math.log10(raw_step))
        for m in (1, 2, 2.5, 5, 10):
            nice = m * mag
            if nice >= raw_step:
                break
        step = max(1, int(round(nice)))
        heatmap_levels = min(64, max(10, int(math.ceil(max_pop / step))))
        bounds = np.arange(heatmap_levels + 1, dtype=float) * step
    else:
        # явное число уровней: границы равномерно от 0 до max_pop
        heatmap_levels = max(2, int(heatmap_levels))
        bounds = np.linspace(0, max(max_pop, 1), heatmap_levels + 1)

    cmap = make_discrete_popularity_cmap(heatmap_levels)
    pop_per_color = float(bounds[1] - bounds[0])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # каждая ячейка закрашивается одним сплошным цветом своего уровня;
    # тонкая рамка того же цвета лишь убирает щели между ячейками
    mesh = ax.pcolormesh(xe, ye, H.T, cmap=cmap, norm=matplotlib.colors.BoundaryNorm(
        bounds, cmap.N), edgecolors="face", linewidths=0.5, rasterized=True)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])

    # Если уровней много, подписи всех границ не влезают — показываем подмножество
    tick_vals = bounds
    if cmap.N > 25:
        step_t = int(math.ceil(cmap.N / 16.0))
        tick_vals = np.concatenate([bounds[::step_t], [bounds[-1]]])
    fmt = mticker.FuncFormatter(
        lambda v, _: f"{v:.10g}".rstrip("0").rstrip("."))
    cbar = fig.colorbar(mesh, ax=ax, ticks=tick_vals, extend="max", format=fmt)
    ppc_txt = "1 кадр" if pop_per_color == 1 else f"{pop_per_color:g} кадров"
    cbar.set_label(f"число кадров (радужная шкала: {cmap.N} разных цветов, "
                   f"каждый цвет = интервал {ppc_txt})")
    cbar.outline.set_visible(True)

    # целевая точка
    ax.plot(DI_target, GI_target, marker="*", ms=18, mfc="red", mec="black",
            ls="none", label=f"цель (DI={DI_target:.3f}, GI={GI_target:.3f})")
    # окно допуска
    rect = plt.Rectangle((DI_target - tol_DI, GI_target - tol_GI),
                         2 * tol_DI, 2 * tol_GI,
                         fill=False, edgecolor="red", lw=2, linestyle="--",
                         label=f"допуск ±({tol_DI}, {tol_GI})")
    ax.add_patch(rect)

    ax.set_xlabel("DI (Dimer Index)")
    ax.set_ylabel("GI (Growth Index)")
    ax.set_title("Распределение кадров траектории по коллективным переменным\n"
                 "(тепловая карта популяций, разноцветная радужная шкала)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"График сохранён: {out_png} (разных цветов: {cmap.N}, "
          f"каждый покрывает интервал популяции {pop_per_color:g})")


# ---------------------------------------------------------------------------
# Timeline: полоса времени всей траектории с выделенными отобранными интервалами
# ---------------------------------------------------------------------------
def merge_selected_intervals(sel_pos, frames, times, dt_frames=None):
    """Склеивает отобранные кадры в непрерывные временные интервалы.

    Два соседних отобранных кадра считаются частью одного интервала, если они
    идут подряд в траектории (разница номеров кадров == stride, по умолчанию 1).
    Границы интервала — середины «зазоров» до/после первого и последнего кадра
    интервала (т.е. интервал покрывает время, в течение которого система
    находилась в целевой области DI/GI).

    Параметры:
        sel_pos   : позиции отобранных кадров в массивах frames/times (возр. порядок)
        frames    : номера всех кадров траектории
        times     : времена всех кадров (пс)
        dt_frames : максимальный шаг между «соседними» кадрами (по умолчанию 1)
    Возвращает список словарей:
        {t_start, t_end, f_start, f_end, n_frames}
    """
    if len(sel_pos) == 0:
        return []
    stride = int(dt_frames) if dt_frames else 1
    intervals = []
    run = [int(sel_pos[0])]
    for pos in list(sel_pos[1:]) + [None]:
        if pos is not None and (int(frames[pos]) - int(frames[run[-1]])) <= stride:
            run.append(int(pos))
            continue
        # закрываем накопленный интервал
        first, last = run[0], run[-1]
        # границы: середина зазора с предыдущим/следующим кадром траектории
        if first - 1 >= 0:
            t_start = 0.5 * (float(times[first - 1]) + float(times[first]))
            f_start = int(frames[first - 1])
        else:
            t_start = float(times[first])
            f_start = int(frames[first])
        if last + 1 < len(frames):
            t_end = 0.5 * (float(times[last]) + float(times[last + 1]))
            f_end = int(frames[last + 1])
        else:
            t_end = float(times[last])
            f_end = int(frames[last])
        intervals.append({
            "t_start": t_start, "t_end": t_end,
            "f_start": f_start, "f_end": f_end,
            "n_frames": len(run),
        })
        if pos is not None:
            run = [int(pos)]
    return intervals


def plot_timeline_static(frames, times, intervals, out_png, title=None):
    """Статическая (PNG) версия timeline средствами matplotlib.

    Серая полоса = всё время траектории; поверх — яркие прямоугольники
    интервалов, когда система была в целевой области DI/GI.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: matplotlib не установлен, пропускаю PNG-timeline.")
        return False

    t0, t1 = float(times.min()), float(times.max())
    fig, ax = plt.subplots(figsize=(12, 2.6))

    # серая полоса — вся траектория
    ax.add_patch(Rectangle((t0, 0.35), t1 - t0, 0.30, facecolor="#b0b0b0",
                           edgecolor="#808080", lw=0.5, zorder=1))
    # яркие интервалы selected
    colors = ["#e4572e", "#f3a712", "#00a5cf", "#7ae582", "#ff4d6d"]
    for k, iv in enumerate(intervals):
        w = max(iv["t_end"] - iv["t_start"], (t1 - t0) / max(len(frames), 1) * 0.9)
        ax.add_patch(Rectangle((iv["t_start"], 0.30), w, 0.40,
                               facecolor=colors[k % len(colors)],
                               edgecolor="black", lw=0.4, zorder=2))
        ax.text(iv["t_start"] + w / 2.0, 0.78, f"#{k+1}", ha="center",
                va="bottom", fontsize=7, color=colors[k % len(colors)], zorder=3)

    ax.set_xlim(t0 - 0.02 * (t1 - t0 + 1e-9), t1 + 0.02 * (t1 - t0 + 1e-9))
    ax.set_ylim(0.15, 1.0)
    ax.get_yaxis().set_visible(False)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Время, пс")
    ax.set_title(title or
                 (f"Полоса времени траектории ({t0:.0f}–{t1:.0f} пс); "
                  f"ярким — интервалы попадания в цель (всего {len(intervals)})"),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Timeline (статический PNG) сохранён: {out_png}")
    return True


def plot_timeline_interactive(frames, times, di_all, gi_all, intervals, out_html,
                              target=None, tol=None, title=None):
    """Интерактивная timeline (HTML, plotly): серая полоса всей траектории,
    яркие интервалы selected реагируют на наведение курсора всплывающей
    подсказкой (время, номера кадров, длительность, число кадров, средние
    DI/GI внутри интервала). Дополнительно — графики DI(t)/GI(t) с окном допуска.

    Возвращает True, если файл создан; False — если plotly недоступен.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: plotly не установлен (pip install plotly), "
              "пропускаю интерактивный HTML-timeline; будет только PNG.")
        return False

    t0, t1 = float(times.min()), float(times.max())
    span = t1 - t0 if t1 > t0 else 1.0

    fig = make_subplots_safe(go)

    # --- ряд 1: серая полоса всей траектории -------------------------------
    fig.add_trace(
        go.Scatter(x=[t0, t1], y=[1, 1], mode="lines",
                   line=dict(color="rgb(176,176,176)", width=22),
                   name="Вся траектория",
                   hovertemplate=(f"<b>Вся траектория</b><br>"
                                  f"{t0:.1f} – {t1:.1f} пс<br>"
                                  f"длительность: {span:.1f} пс<br>"
                                  f"кадров: {len(frames)}<extra></extra>")),
        row=1, col=1)

    # --- ряд 1: яркие интервалы selected ------------------------------------
    colors = ["#e4572e", "#f3a712", "#00a5cf", "#7ae582", "#ff4d6d", "#b892ff"]
    for k, iv in enumerate(intervals):
        idx = np.nonzero((frames >= iv["f_start"]) & (frames <= iv["f_end"]))[0]
        # реальные (попадные) границы по времени выбранных кадров
        ts_sel = times[idx]
        di_sel = di_all[idx] if len(idx) else np.array([np.nan])
        gi_sel = gi_all[idx] if len(idx) else np.array([np.nan])
        dur = iv["t_end"] - iv["t_start"]
        fig.add_trace(
            go.Scatter(x=[iv["t_start"], iv["t_end"]], y=[1, 1], mode="lines",
                       line=dict(color=colors[k % len(colors)], width=22),
                       name=f"Интервал #{k+1}",
                       showlegend=False,
                       hovertemplate=(
                           f"<b>Интервал #{k+1} (selected)</b><br>"
                           f"Время: {iv['t_start']:.1f} – {iv['t_end']:.1f} пс "
                           f"(Δ={dur:.1f} пс)<br>"
                           f"Кадры: {iv['f_start']} – {iv['f_end']} "
                           f"({len(ts_sel)} попавших)<br>"
                           f"DI: {di_sel.min():.3f} … {di_sel.max():.3f} "
                           f"(средн. {np.nanmean(di_sel):.3f})<br>"
                           f"GI: {gi_sel.min():.3f} … {gi_sel.max():.3f} "
                           f"(средн. {np.nanmean(gi_sel):.3f})"
                           "<extra></extra>")),
            row=1, col=1)

    # --- ряд 2: DI(t), ряд 3: GI(t) — контекст ------------------------------
    fig.add_trace(go.Scatter(x=times, y=di_all, mode="lines", name="DI(t)",
                             line=dict(color="#3366cc", width=1),
                             hovertemplate="t=%{x:.1f} пс<br>DI=%{y:.4f}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=gi_all, mode="lines", name="GI(t)",
                             line=dict(color="#cc3311", width=1),
                             hovertemplate="t=%{x:.1f} пс<br>GI=%{y:.4f}<extra></extra>"),
                  row=3, col=1)

    # окно допуска / целевая точка на рядах DI/GI
    if target is not None:
        DI_t, GI_t = target
        hDI = dict(row=2, col=1); hGI = dict(row=3, col=1)
        if tol is not None:
            tDI, tGI = tol
            fig.add_hrect(y0=DI_t - tDI, y1=DI_t + tDI, line=dict(color="red", width=1, dash="dot"),
                          fillcolor="rgba(255,0,0,0.08)", **hDI)
            fig.add_hrect(y0=GI_t - tGI, y1=GI_t + tGI, line=dict(color="red", width=1, dash="dot"),
                          fillcolor="rgba(255,0,0,0.08)", **hGI)
        fig.add_hline(y=DI_t, line=dict(color="red", width=1), **hDI)
        fig.add_hline(y=GI_t, line=dict(color="red", width=1), **hGI)

    # подсветка интервалов на рядах DI/GI вертикальными shapes
    for k, iv in enumerate(intervals):
        for r in (2, 3):
            fig.add_vrect(x0=iv["t_start"], x1=iv["t_end"], row=r, col=1,
                          fillcolor=colors[k % len(colors)], opacity=0.12,
                          line_width=0)

    fig.update_xaxes(range=[t0 - 0.02 * span, t1 + 0.02 * span], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0.6, 1.4], row=1, col=1)
    fig.update_xaxes(title_text="Время, пс", row=3, col=1)
    fig.update_yaxes(title_text="DI", row=2, col=1)
    fig.update_yaxes(title_text="GI", row=3, col=1)
    fig.update_layout(
        height=620,
        template="plotly_white",
        title=title or ("Полоса времени траектории: серым — всё время, "
                        "ярким — кадры, попавшие в selected (наведите курсор)"),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="closest",
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Timeline (интерактивный HTML) сохранён: {out_html}")
    return True


def make_subplots_safe(go):
    """Фигура из 3 строк (timeline / DI(t) / GI(t)) с общими осями X."""
    from plotly.subplots import make_subplots
    return make_subplots(rows=3, cols=1, shared_xaxes=True,
                         row_heights=[0.18, 0.41, 0.41], vertical_spacing=0.08)


# ---------------------------------------------------------------------------
# CSV -> heatmap (без пересчёта траектории)
# ---------------------------------------------------------------------------
def load_di_gi_from_csv(path):
    """Читает out_csv (колонки frame,time_ps,DI,GI) и возвращает массивы DI, GI.

    Строчка заголовка пропускается; пустые строки игнорируются.
    """
    di_vals, gi_vals = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                raise ValueError(f"Строка '{line}' в {path}: ожидалось 4 колонки "
                                 "(frame,time_ps,DI,GI). Возможно, это не CSV из этой программы?")
            try:
                di_vals.append(float(parts[2]))
                gi_vals.append(float(parts[3]))
            except ValueError:
                continue  # строка заголовка ("DI"/"GI" не парсятся как float)
    if not di_vals:
        raise ValueError(f"В файле {path} нет ни одной строки с числовыми DI/GI.")
    return np.asarray(di_vals), np.asarray(gi_vals)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Отбор кадров МД-траектории по значениям коллективных переменных DI и GI.\n"
                    "Режим пересчёта: укажите --gro/--xtc/--ndx_O/--ndx_C.\n"
                    "Режим heatmap из CSV: укажите --from_csv (тогда --DI/--GI необязательны).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gro",
                   help="Файл топологии (.gro или .pdb): массы, порядок атомов, бокс.")
    p.add_argument("--xtc",
                   help="Траектория (.xtc).")
    p.add_argument("--ndx_O",
                   help="Index-файл с группами centralO (cal1_centralO ... cal6_centralO).")
    p.add_argument("--ndx_C",
                   help="Index-файл с группами tailC (cal1_tailC ... cal6_tailC).")
    p.add_argument("--from_csv", metavar="CSV",
                   help="Построить только 2D-гистограмму (DI, GI) по уже рассчитанному CSV "
                        "(колонки frame,time_ps,DI,GI) — без загрузки и пересчёта траектории. "
                        "В этом режиме результаты отбора НЕ перезаписываются.")
    p.add_argument("--DI", type=float, help="Целевое значение Dimer Index.")
    p.add_argument("--GI", type=float, help="Целевое значение Growth Index.")
    p.add_argument("--tol_DI", type=float, default=0.2, help="Допуск по DI.")
    p.add_argument("--tol_GI", type=float, default=0.3, help="Допуск по GI.")
    p.add_argument("--out_xtc", help="Выходная траектория с отобранными кадрами.")
    p.add_argument("--out_csv", help="CSV со значениями DI/GI для всех кадров.")
    p.add_argument("--plot", action="store_true",
                   help="Построить 2D-гистограмму (DI, GI) и сохранить в --out_png.")
    p.add_argument("--out_png", default="DI_GI_hist.png",
                   help="Имя файла графика при использовании --plot.")
    p.add_argument("--heatmap_levels", type=int, metavar="N", default=None,
                   help="Число разных цветов радужной шкалы тепловой карты "
                        "(каждый цвет покрывает свой интервал популяции). "
                        "По умолчанию подбирается автоматически (до 64).")
    p.add_argument("--out_timeline_html", metavar="HTML", default=None,
                   help="Интерактивная полоса времени (plotly HTML): серая полоса — "
                        "всё время траектории, яркие интервалы — кадры из selected; "
                        "при наведении курсора показывается информация об интервале. "
                        "По умолчанию: <out_csv без расширения>_timeline.html. "
                        "Пустая строка \"\" — не создавать.")
    p.add_argument("--out_timeline_png", metavar="PNG", default=None,
                   help="Статическая (PNG) версия полосы времени (matplotlib). "
                        "По умолчанию: <out_csv без расширения>_timeline.png. "
                        "Пустая строка \"\" — не создавать.")
    p.add_argument("--no_timeline", action="store_true",
                   help="Не строить полосу времени вообще (ни HTML, ни PNG).")
    p.add_argument("--top_n", type=int, default=None,
                   help="Вместо окна допуска выбрать N кадров, ближайших к (DI, GI) "
                        "по евклидову расстоянию.")
    p.add_argument("--progress", type=int, default=1000,
                   help="Частота вывода прогресса (каждые N кадров).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Режим 2: heatmap по уже рассчитанному CSV (без пересчёта) ----------
    if args.from_csv:
        if not os.path.isfile(args.from_csv):
            sys.exit(f"ОШИБКА: CSV-файл не найден: {args.from_csv}")
        try:
            di_all, gi_all = load_di_gi_from_csv(args.from_csv)
        except ValueError as e:
            sys.exit(f"ОШИБКА: {e}")
        print(f"Загружено кадров из CSV: {len(di_all)} ({args.from_csv})")
        print(f"DI: min={di_all.min():.3f}, max={di_all.max():.3f}; "
              f"GI: min={gi_all.min():.3f}, max={gi_all.max():.3f}")
        DI_target = args.DI if args.DI is not None else float(np.median(di_all))
        GI_target = args.GI if args.GI is not None else float(np.median(gi_all))
        if args.DI is None or args.GI is None:
            print("ПРЕДУПРЕЖДЕНИЕ: --DI/--GI не заданы — целевая точка на графике "
                  "поставлена в медиану распределения.")
        plot_histogram(di_all, gi_all, DI_target, GI_target,
                       args.tol_DI, args.tol_GI, args.out_png,
                       heatmap_levels=args.heatmap_levels)
        return

    # --- Режим 1: полный пересчёт по траектории ------------------------------
    missing = [name for name, val in (("--gro", args.gro), ("--xtc", args.xtc),
                                      ("--ndx_O", args.ndx_O), ("--ndx_C", args.ndx_C),
                                      ("--out_xtc", args.out_xtc), ("--out_csv", args.out_csv))
               if not val]
    if missing:
        sys.exit("ОШИБКА: для расчёта по траектории обязательны аргументы: "
                 + ", ".join(missing) + "\n(если нужно только перестроить график по CSV — "
                 "используйте --from_csv)")
    if args.DI is None or args.GI is None:
        sys.exit("ОШИБКА: задайте целевые значения --DI и --GI.")

    # --- 1. Загрузка топологии и траектории через MDAnalysis ---------------
    for f in (args.gro, args.xtc, args.ndx_O, args.ndx_C):
        if not os.path.isfile(f):
            sys.exit(f"ОШИБКА: входной файл не найден: {f}")

    print(f"Загрузка: топология={args.gro}, траектория={args.xtc}")
    try:
        u = mda.Universe(args.gro, args.xtc)
    except Exception as e:
        sys.exit(f"ОШИБКА: не удалось загрузить систему: {e}")

    natoms = u.atoms.n_atoms
    expected_natoms = N_MONOMERS * ATOMS_PER_MONOMER  # 456
    if natoms != expected_natoms:
        print(f"ПРЕДУПРЕЖДЕНИЕ: в топологии {natoms} атомов вместо ожидаемых {expected_natoms}. "
              f"Индексные группы всё равно читаются напрямую из .ndx; убедитесь, что "
              f"блоки мономеров соответствуют 1-76, 77-152, ..., 381-456.")

    # --- 2. Чтение index-файлов --------------------------------------------
    groups_O = read_ndx(args.ndx_O)
    groups_C = read_ndx(args.ndx_C)

    o_idx, c_idx = [], []
    for m in range(1, N_MONOMERS + 1):
        name_o = f"cal{m}_centralO"
        name_c = f"cal{m}_tailC"
        try:
            o_idx.append(get_group_indices(groups_O, name_o, args.ndx_O, natoms))
            c_idx.append(get_group_indices(groups_C, name_c, args.ndx_C, natoms))
        except (KeyError, ValueError) as e:
            sys.exit(f"ОШИБКА: {e}")

    print("Прочитаны группы:")
    for m in range(N_MONOMERS):
        print(f"  cal{m+1}: centralO ({len(o_idx[m])} ат.) = {o_idx[m]+1}, "
              f"tailC ({len(c_idx[m])} ат.) = {c_idx[m]+1}  [показаны 1-based]")

    # --- 3. Проход по траектории: расчёт DI/GI для каждого кадра -----------
    frames, times, di_all, gi_all = analyze_trajectory(
        u, o_idx, c_idx, progress_every=args.progress)

    if len(frames) == 0:
        sys.exit("ОШИБКА: в траектории нет ни одного кадра.")

    # --- 4. Запись CSV со всеми кадрами -------------------------------------
    with open(args.out_csv, "w") as fh:
        fh.write("frame,time_ps,DI,GI\n")
        for fr, t, di, gi in zip(frames, times, di_all, gi_all):
            fh.write(f"{fr},{t:.4f},{di:.6f},{gi:.6f}\n")
    print(f"Таблица всех кадров записана: {args.out_csv}")

    # --- 5–6. Отбор кадров ---------------------------------------------------
    if args.top_n is not None:
        # режим: N ближайших кадров по евклидову расстоянию в плоскости (DI, GI)
        dist = np.hypot(di_all - args.DI, gi_all - args.GI)
        n_sel = min(args.top_n, len(frames))
        sel_pos = np.argsort(dist)[:n_sel]
        sel_pos = np.sort(sel_pos)  # сохраняем порядок следования в траектории
        mode = f"top_n={n_sel} ближайших кадров"
    else:
        mask = (np.abs(di_all - args.DI) <= args.tol_DI) & \
               (np.abs(gi_all - args.GI) <= args.tol_GI)
        sel_pos = np.nonzero(mask)[0]
        mode = (f"окно допуска: |DI-{args.DI}|<={args.tol_DI}, "
                f"|GI-{args.GI}|<={args.tol_GI}")

    nearest_pos = int(np.argmin(np.hypot(di_all - args.DI, gi_all - args.GI)))
    d_near = float(np.hypot(di_all[nearest_pos] - args.DI, gi_all[nearest_pos] - args.GI))

    print("\n--- Отчёт ---")
    print(f"Режим отбора: {mode}")
    print(f"Всего кадров: {len(frames)}")
    print(f"DI по траектории: min={di_all.min():.4f}, max={di_all.max():.4f}")
    print(f"GI по траектории: min={gi_all.min():.4f}, max={gi_all.max():.4f}")
    print(f"Ближайший кадр к цели: frame={frames[nearest_pos]} (time={times[nearest_pos]:.1f} ps), "
          f"DI={di_all[nearest_pos]:.4f}, GI={gi_all[nearest_pos]:.4f}, "
          f"евклидово расстояние={d_near:.4f}")

    if len(sel_pos) == 0:
        print("\nПРЕДУПРЕЖДЕНИЕ: ни один кадр не попал в заданное окно допуска!")
        print(f"  Самый близкий кадр: frame={frames[nearest_pos]}, "
              f"time={times[nearest_pos]:.1f} ps, DI={di_all[nearest_pos]:.4f}, "
              f"GI={gi_all[nearest_pos]:.4f}.")
        print("  Совет: увеличьте --tol_DI/--tol_GI или задайте --top_n 1, "
              "чтобы явно сохранить ближайший кадр.")
        # выходной xtc при пустом окне не создаём — иначе это битый файл
        print(f"  Файл {args.out_xtc} НЕ создан (нет отобранных кадров).")
    else:
        sel_frames = frames[sel_pos]
        print(f"\nОтобрано кадров: {len(sel_pos)}")
        preview = ", ".join(str(int(f)) for f in sel_frames[:50])
        if len(sel_frames) > 50:
            preview += ", ..."
        print(f"Номера кадров: {preview}")

        # --- 7. Сохранение отобранных кадров в XTC --------------------------
        # ВНИМАНИЕ: XTC содержит только координаты (и бокс), без имён атомов.
        # ChimeraX/VMD подставляют к ним топологию из .gro/.pdb — если открыть
        # selected.xtc вместе с solv-топологией (14451 ат.), получится ошибка
        # "structure has N atoms whereas the coordinates are for 456 atoms".
        # Поэтому рядом с XTC всегда пишем согласованный ref-frame .gro на ТЕ
        # же самые атомы и в том же порядке (универсальный способ для любого
        # набора атомов: берём Universe, созданный программой по --gro).
        w = mda.Writer(args.out_xtc, u.atoms.n_atoms)
        count = 0
        for pos in sel_pos:
            ts = u.trajectory[int(frames[pos])]   # переход к нужному кадру
            w.write(u.atoms)
            count += 1
        w.close()
        print(f"Сохранено {count} кадров -> {args.out_xtc}")

        base, _ = os.path.splitext(args.out_xtc)
        gro_out = base + ".gro"
        first_sel_frame = int(frames[sel_pos[0]])
        u.trajectory[first_sel_frame]
        try:
            u.atoms.write(gro_out)
            print(f"Референсная топология ({u.atoms.n_atoms} ат., кадр {first_sel_frame}) "
                  f"записана -> {gro_out}")
            print("Открывайте в ChimeraX ЭТУ пару файлов, например:\n"
                  f"    open {gro_out}\n"
                  f"    movie read {args.out_xtc} start {first_sel_frame}")
        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось записать {gro_out}: {e}")

    # --- опциональный график ------------------------------------------------
    if args.plot:
        plot_histogram(di_all, gi_all, args.DI, args.GI, args.tol_DI, args.tol_GI,
                       args.out_png, heatmap_levels=args.heatmap_levels)

    # --- полоса времени (timeline): серая — вся траектория, яркие — selected --
    if not args.no_timeline:
        intervals = merge_selected_intervals(sel_pos, frames, times)
        base_csv, _ = os.path.splitext(args.out_csv)
        html_out = args.out_timeline_html if args.out_timeline_html is not None \
            else base_csv + "_timeline.html"
        png_out = args.out_timeline_png if args.out_timeline_png is not None \
            else base_csv + "_timeline.png"
        tl_title = (f"Полоса времени траектории; цель: DI={args.DI:.3f}, GI={args.GI:.3f}; "
                    f"отобрано кадров: {len(sel_pos)}")
        if len(intervals) == 0:
            print("Примечание: ни один кадр не попал в окно допуска — "
                  "timeline будет построена без ярких интервалов.")
        if html_out:
            plot_timeline_interactive(frames, times, di_all, gi_all, intervals, html_out,
                                      target=(args.DI, args.GI),
                                      tol=(args.tol_DI, args.tol_GI),
                                      title=tl_title)
        if png_out:
            plot_timeline_static(frames, times, intervals, png_out, title=tl_title)

    print("\nГотово.")


if __name__ == "__main__":
    main()
