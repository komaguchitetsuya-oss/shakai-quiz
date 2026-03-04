"""
PDF問題集のオレンジ文字を検出し、マスク画像と座標データを生成するスクリプト。

マルチクイズ対応:
- data/{quiz_id}/pages/ に画像とregions.jsonを出力
- split/data/{quiz_id}/pages/ に左右分割版を出力
- quizzes.json（トップレベル）にクイズ一覧を更新

使い方:
  python3 process_pages.py --pdf ~/Downloads/XXX.pdf --quiz-id 630-02 --title "630-02 植物の働き①"
  python3 process_pages.py --pdf ~/Downloads/XXX.pdf --quiz-id 630-02 --title "630-02 植物の働き①" --skip-cover
"""

import argparse
import json
import os
import re
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(__file__)

# オレンジ検出パラメータ
BLOCK_SIZE = 5         # ピクセルをブロック単位でグループ化（小さい＝細かく分離）
DILATE_BLOCKS = 2      # 隣接ブロックをマージするための膨張量（同一単語内の文字をつなげる）
PADDING = 3            # 領域のパディング（ピクセル）
MIN_REGION_SIZE = 12   # これより小さい領域はノイズとしてスキップ
LINE_Y_TOLERANCE = 20  # この範囲内の領域は「同じ行」として左→右でソート


def detect_orange_mask(img_array):
    """オレンジ色のピクセルを検出して boolean mask を返す"""
    r = img_array[:, :, 0].astype(np.int16)
    g = img_array[:, :, 1].astype(np.int16)
    b = img_array[:, :, 2].astype(np.int16)

    mask = (
        (r > 170) &
        ((r - b) > 35) &
        ((r - g) > 10) &
        ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 35)
    )
    return mask


def find_regions(orange_mask, block_size=BLOCK_SIZE, dilate=DILATE_BLOCKS, page_width=None):
    """オレンジマスクからブロック単位で連結領域を見つける"""
    h, w = orange_mask.shape

    bh = (h + block_size - 1) // block_size
    bw = (w + block_size - 1) // block_size
    block_grid = np.zeros((bh, bw), dtype=bool)

    for by in range(bh):
        for bx in range(bw):
            y0 = by * block_size
            y1 = min(y0 + block_size, h)
            x0 = bx * block_size
            x1 = min(x0 + block_size, w)
            if orange_mask[y0:y1, x0:x1].any():
                block_grid[by, bx] = True

    # 膨張（水平方向のみ）
    if dilate > 0:
        dilated = block_grid.copy()
        for _ in range(dilate):
            new = dilated.copy()
            for by in range(bh):
                for bx in range(bw):
                    if dilated[by, bx]:
                        for dx in range(-1, 2):
                            nx = bx + dx
                            if 0 <= nx < bw:
                                new[by, nx] = True
            dilated = new
        block_grid = dilated

    # 連結成分検出（4方向フラッドフィル）
    labels = np.zeros((bh, bw), dtype=int)
    label_id = 0

    for by in range(bh):
        for bx in range(bw):
            if block_grid[by, bx] and labels[by, bx] == 0:
                label_id += 1
                stack = [(by, bx)]
                while stack:
                    cy, cx = stack.pop()
                    if (0 <= cy < bh and 0 <= cx < bw and
                            block_grid[cy, cx] and labels[cy, cx] == 0):
                        labels[cy, cx] = label_id
                        stack.extend([(cy-1, cx), (cy+1, cx),
                                      (cy, cx-1), (cy, cx+1)])

    regions = []
    for lid in range(1, label_id + 1):
        ys, xs = np.where(labels == lid)
        y_min = int(ys.min()) * block_size
        y_max = min(int(ys.max() + 1) * block_size, orange_mask.shape[0])
        x_min = int(xs.min()) * block_size
        x_max = min(int(xs.max() + 1) * block_size, orange_mask.shape[1])

        region_mask = orange_mask[y_min:y_max, x_min:x_max]
        if not region_mask.any():
            continue
        oy, ox = np.where(region_mask)
        ry_min = y_min + int(oy.min()) - PADDING
        ry_max = y_min + int(oy.max()) + PADDING
        rx_min = x_min + int(ox.min()) - PADDING
        rx_max = x_min + int(ox.max()) + PADDING

        ry_min = max(0, ry_min)
        ry_max = min(orange_mask.shape[0], ry_max)
        rx_min = max(0, rx_min)
        rx_max = min(orange_mask.shape[1], rx_max)

        if (rx_max - rx_min) < MIN_REGION_SIZE or (ry_max - ry_min) < MIN_REGION_SIZE:
            continue

        regions.append({
            "x": rx_min, "y": ry_min,
            "w": rx_max - rx_min, "h": ry_max - ry_min,
        })

    mid_x = (page_width or orange_mask.shape[1]) // 2
    regions.sort(key=lambda r: (
        0 if r["x"] + r["w"] / 2 < mid_x else 1,
        r["y"] // LINE_Y_TOLERANCE,
        r["x"]
    ))

    return regions


def dilate_mask(mask, radius=2):
    """マスクを数ピクセル膨張させてアンチエイリアスのエッジも消す"""
    h, w = mask.shape
    dilated = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            sy = max(0, dy)
            ey = min(h, h + dy)
            sx = max(0, dx)
            ex = min(w, w + dx)
            shifted[sy:ey, sx:ex] = mask[sy - dy:ey - dy, sx - dx:ex - dx]
            dilated |= shifted
    return dilated


def create_masked_image(img_array, orange_mask):
    """オレンジピクセル＋周辺エッジを白で隠した画像を作成"""
    expanded = dilate_mask(orange_mask, radius=2)
    masked = img_array.copy()
    masked[expanded] = 255
    return masked


def enhance_image(img):
    """スキャン画像のコントラストとシャープネスを向上させて可読性を上げる"""
    img = ImageEnhance.Contrast(img).enhance(1.65)
    img = ImageEnhance.Sharpness(img).enhance(2.1)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=75, threshold=2))
    return img


def find_gray_labels(img_array, block_size=10, min_blocks=30):
    """灰色背景の吹き出し領域をラベリング（大きな灰色領域のみ）"""
    h, w = img_array.shape[:2]
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    avg = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    gray_mask = (avg > 160) & (avg < 235) & (sat < 30)

    bh = (h + block_size - 1) // block_size
    bw = (w + block_size - 1) // block_size
    block_grid = np.zeros((bh, bw), dtype=bool)

    for by in range(bh):
        for bx in range(bw):
            y0 = by * block_size
            y1 = min(y0 + block_size, h)
            x0 = bx * block_size
            x1 = min(x0 + block_size, w)
            block_area = (y1 - y0) * (x1 - x0)
            if block_area > 0 and gray_mask[y0:y1, x0:x1].sum() > block_area * 0.4:
                block_grid[by, bx] = True

    labels = np.zeros((bh, bw), dtype=int)
    label_id = 0
    label_counts = {}

    for by in range(bh):
        for bx in range(bw):
            if block_grid[by, bx] and labels[by, bx] == 0:
                label_id += 1
                count = 0
                stack = [(by, bx)]
                while stack:
                    cy, cx = stack.pop()
                    if (0 <= cy < bh and 0 <= cx < bw and
                            block_grid[cy, cx] and labels[cy, cx] == 0):
                        labels[cy, cx] = label_id
                        count += 1
                        stack.extend([(cy-1, cx), (cy+1, cx),
                                      (cy, cx-1), (cy, cx+1)])
                label_counts[label_id] = count

    for lid, count in label_counts.items():
        if count < min_blocks:
            labels[labels == lid] = 0

    return labels, block_size


def merge_regions_in_bubbles(regions, gray_labels, block_size):
    """灰色背景の吹き出し内にある複数の領域を1つにマージ"""
    bh, bw = gray_labels.shape
    groups = {}
    ungrouped = []

    for i, r in enumerate(regions):
        cx = r["x"] + r["w"] // 2
        cy = r["y"] + r["h"] // 2
        by = min(cy // block_size, bh - 1)
        bx = min(cx // block_size, bw - 1)
        label = int(gray_labels[by, bx])
        if label > 0:
            groups.setdefault(label, []).append(i)
        else:
            ungrouped.append(i)

    merged = [regions[i] for i in ungrouped]

    for label, indices in groups.items():
        if len(indices) == 1:
            merged.append(regions[indices[0]])
        else:
            group = [regions[i] for i in indices]
            x_min = min(r["x"] for r in group)
            y_min = min(r["y"] for r in group)
            x_max = max(r["x"] + r["w"] for r in group)
            y_max = max(r["y"] + r["h"] for r in group)
            merged.append({
                "x": x_min, "y": y_min,
                "w": x_max - x_min, "h": y_max - y_min,
            })

    return merged


def auto_crop(img_array, margin=10, threshold=240):
    """白い余白を検出してトリミング"""
    gray = np.mean(img_array[:, :, :3], axis=2)
    non_white = gray < threshold
    row_has_content = non_white.any(axis=1)
    col_has_content = non_white.any(axis=0)

    if not row_has_content.any() or not col_has_content.any():
        return img_array, 0, 0

    rows = np.where(row_has_content)[0]
    cols = np.where(col_has_content)[0]

    y_min = max(0, rows[0] - margin)
    y_max = min(img_array.shape[0], rows[-1] + margin)
    x_min = max(0, cols[0] - margin)
    x_max = min(img_array.shape[1], cols[-1] + margin)

    return img_array[y_min:y_max, x_min:x_max], x_min, y_min


def auto_rotate(img):
    """スキャンPDFを日本語横書きとして読めるよう90°時計回りに回転"""
    return img.transpose(Image.ROTATE_270)


def extract_title_from_pdf(doc):
    """PDFの表紙からタイトルを抽出（テキストレイヤーがある場合）"""
    if len(doc) > 0:
        text = doc[0].get_text()
        if text.strip():
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in lines:
                if re.match(r'\d{3}-\d{2}', line):
                    return line
    return None


def get_quiz_id_from_title(title):
    """タイトルからクイズID（例: "630-02"）を抽出"""
    match = re.match(r'(\d{3}-\d{2})', title)
    return match.group(1) if match else None


def update_quizzes_json(quiz_id, title, pages_data, target_dir):
    """トップレベルの quizzes.json にクイズ情報を追加/更新"""
    json_path = os.path.join(target_dir, "quizzes.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"quizzes": []}

    total_regions = sum(len(p["regions"]) for p in pages_data)

    data["quizzes"] = [q for q in data["quizzes"] if q["id"] != quiz_id]
    data["quizzes"].append({
        "id": quiz_id,
        "title": title,
        "pageCount": len(pages_data),
        "totalRegions": total_regions,
    })
    data["quizzes"].sort(key=lambda q: q["id"])

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {json_path}: {quiz_id} ({title}) - {len(pages_data)} pages, {total_regions} regions")


def process_pdf(pdf_path, quiz_id, title, skip_cover=False):
    """PDFの全ページを処理"""
    pages_dir = os.path.join(BASE_DIR, "data", quiz_id, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    all_pages = []
    start_page = 1 if skip_cover else 0
    output_num = 0

    for page_num in range(start_page, len(doc)):
        output_num += 1
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        w, h = pix.width, pix.height

        raw_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)
        if pix.n == 4:
            raw_array = raw_array[:, :, :3]

        pil_img = Image.fromarray(raw_array)
        pil_img = auto_rotate(pil_img)
        img_array = np.array(pil_img)
        h, w = img_array.shape[:2]

        orange_mask = detect_orange_mask(img_array)
        orange_count = orange_mask.sum()

        regions = find_regions(orange_mask, page_width=w)

        gray_labels, gb_bs = find_gray_labels(img_array)
        regions = merge_regions_in_bubbles(regions, gray_labels, gb_bs)
        mid_x = w // 2
        regions.sort(key=lambda r: (
            0 if r["x"] + r["w"] / 2 < mid_x else 1,
            r["y"] // LINE_Y_TOLERANCE,
            r["x"]
        ))

        img_array, crop_x, crop_y = auto_crop(img_array)
        orange_mask = orange_mask[crop_y:crop_y + img_array.shape[0],
                                  crop_x:crop_x + img_array.shape[1]]
        h, w = img_array.shape[:2]

        for r in regions:
            r["x"] -= crop_x
            r["y"] -= crop_y

        pil_img = Image.fromarray(img_array)
        enhanced_img = enhance_image(pil_img)
        enhanced_array = np.array(enhanced_img)

        orig_path = os.path.join(pages_dir, f"page{output_num}.png")
        enhanced_img.save(orig_path, optimize=True)

        print(f"Page {output_num} (PDF page {page_num + 1}): {w}x{h}")
        print(f"  Orange pixels: {orange_count}")
        print(f"  Regions found: {len(regions)}")

        masked = create_masked_image(enhanced_array, orange_mask)
        masked_img = Image.fromarray(masked)
        masked_path = os.path.join(pages_dir, f"page{output_num}_masked.png")
        masked_img.save(masked_path, optimize=True)

        for i, r in enumerate(regions):
            r["id"] = f"{quiz_id}-p{output_num}-r{i + 1}"

        all_pages.append({
            "page": output_num,
            "width": w,
            "height": h,
            "regions": regions,
        })

        for i, r in enumerate(regions):
            print(f"    Region {i+1}: x={r['x']}, y={r['y']}, w={r['w']}, h={r['h']}")

    doc.close()

    json_path = os.path.join(pages_dir, "regions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "id": quiz_id,
            "title": title,
            "pages": all_pages
        }, f, ensure_ascii=False, indent=2)

    update_quizzes_json(quiz_id, title, all_pages, BASE_DIR)

    print(f"\nDone! Regions saved to {json_path}")
    return all_pages


def generate_split_pages(full_pages_data, quiz_id, title):
    """各ページを左右に分割して split/data/{quiz_id}/pages/ に保存"""
    source_dir = os.path.join(BASE_DIR, "data", quiz_id, "pages")
    split_dir = os.path.join(BASE_DIR, "split", "data", quiz_id, "pages")
    os.makedirs(split_dir, exist_ok=True)

    split_pages = []
    split_num = 0

    for page_data in full_pages_data:
        pn = page_data["page"]
        orig_w = page_data["width"]
        mid_x = orig_w // 2

        orig = np.array(Image.open(os.path.join(source_dir, f"page{pn}.png")))
        masked = np.array(Image.open(os.path.join(source_dir, f"page{pn}_masked.png")))

        for side, label in [("left", "L"), ("right", "R")]:
            split_num += 1
            if side == "left":
                x_start, x_end = 0, mid_x
            else:
                x_start, x_end = mid_x, orig_w

            orig_half = orig[:, x_start:x_end]
            masked_half = masked[:, x_start:x_end]
            sh, sw = orig_half.shape[:2]

            Image.fromarray(orig_half).save(
                os.path.join(split_dir, f"page{split_num}.png"), optimize=True)
            Image.fromarray(masked_half).save(
                os.path.join(split_dir, f"page{split_num}_masked.png"), optimize=True)

            half_regions = []
            for r in page_data["regions"]:
                center_x = r["x"] + r["w"] / 2
                if side == "left" and center_x < mid_x:
                    half_regions.append({
                        "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
                        "id": f"{quiz_id}-s{split_num}-r{len(half_regions) + 1}",
                    })
                elif side == "right" and center_x >= mid_x:
                    half_regions.append({
                        "x": r["x"] - x_start, "y": r["y"], "w": r["w"], "h": r["h"],
                        "id": f"{quiz_id}-s{split_num}-r{len(half_regions) + 1}",
                    })

            half_regions.sort(key=lambda r: (r["y"] // LINE_Y_TOLERANCE, r["x"]))

            split_pages.append({
                "page": split_num,
                "label": f"P{pn}{label}",
                "width": sw,
                "height": sh,
                "regions": half_regions,
            })

            print(f"  Split {split_num} (P{pn}{label}): {sw}x{sh}, {len(half_regions)} regions")

    json_path = os.path.join(split_dir, "regions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "id": quiz_id,
            "title": title,
            "pages": split_pages
        }, f, ensure_ascii=False, indent=2)

    split_base = os.path.join(BASE_DIR, "split")
    update_quizzes_json(quiz_id, title, split_pages, split_base)

    print(f"\nSplit pages saved to {split_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF問題集からクイズデータを生成")
    parser.add_argument("--pdf",
                        default=os.path.expanduser("~/Downloads/20260302173326_001.pdf"),
                        help="PDFファイルのパス")
    parser.add_argument("--quiz-id", default=None, help="クイズID (例: 630-02)")
    parser.add_argument("--title", default=None, help="クイズタイトル (例: '630-02 植物の働き①')")
    parser.add_argument("--skip-cover", action="store_true", help="表紙（1ページ目）をスキップ")
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    auto_title = extract_title_from_pdf(doc)
    doc.close()

    title = args.title or auto_title or os.path.splitext(os.path.basename(args.pdf))[0]
    quiz_id = args.quiz_id or get_quiz_id_from_title(title) or "quiz-1"

    print(f"Quiz ID: {quiz_id}")
    print(f"Title: {title}")
    print(f"PDF: {args.pdf}")
    print(f"Skip cover: {args.skip_cover}")
    print()

    full_data = process_pdf(args.pdf, quiz_id, title, args.skip_cover)
    print("\n--- Generating split pages ---")
    generate_split_pages(full_data, quiz_id, title)
