# -*- coding: utf-8 -*-
"""
system.py

Pipeline:
DIRECT:
    question -> TF-IDF toàn corpus -> Top-K chunk -> ViT5

REFERENTIAL:
    image -> EfficientNet-B0 -> FAISS Top-K image
          -> lọc image threshold
          -> danh sách relic_id không trùng
    question + candidate relic_ids
          -> TF-IDF trong các relic ứng viên
          -> lọc text threshold
          -> Top-K chunk
          -> Top-1 context
          -> ViT5

Không có giao diện. Dùng các hàm Python để test từng bước.
"""

import json
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import ipywidgets as widgets
from IPython.display import display, clear_output


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")

QUESTION_VECTORIZER_PATH = (
    BASE_DIR / "question_classifier" / "tfidf_vectorizer.joblib"
)
QUESTION_MODEL_PATH = (
    BASE_DIR / "question_classifier" / "logistic_regression.joblib"
)

FAISS_INDEX_PATH = (
    BASE_DIR / "faiss" / "image_index.faiss"
)
IMAGE_METADATA_PATH = (
    BASE_DIR / "faiss" / "image_metadata.json"
)
EFFICIENTNET_PATH = (
    BASE_DIR / "efficientnet_b0.pth"
)

TFIDF_VECTORIZER_PATH = (
    BASE_DIR / "tfidf" / "tfidf_vectorizer.joblib"
)
CORPUS_VECTORS_PATH = (
    BASE_DIR / "tfidf" / "corpus_vectors.npz"
)
CORPUS_PATH = (
    BASE_DIR / "tfidf" / "corpus.json"
)

VIT5_MODEL_NAME = "pdlicht04/vit5"

DIRECT_LABEL = 0
REFERENTIAL_LABEL = 1

DEFAULT_TOP_K_IMAGE = 1
DEFAULT_TOP_K_CHUNK = 1

DEFAULT_IMAGE_THRESHOLD = 0.10
DEFAULT_TEXT_THRESHOLD = 0.01

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GLOBAL OBJECTS
# ============================================================

question_vectorizer = None
question_classifier = None

image_model = None
image_preprocess = None
faiss_index = None
image_metadata = None

tfidf_vectorizer = None
corpus_vectors = None
corpus = None

vit5_tokenizer = None
vit5_model = None

SYSTEM_LOADED = False


# ============================================================
# UTILS
# ============================================================

def check_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}"
        )


# ============================================================
# 1. QUESTION CLASSIFIER
# ============================================================

def load_question_classifier():
    global question_vectorizer
    global question_classifier

    check_file(
        QUESTION_VECTORIZER_PATH
    )
    check_file(
        QUESTION_MODEL_PATH
    )

    question_vectorizer = joblib.load(
        QUESTION_VECTORIZER_PATH
    )

    question_classifier = joblib.load(
        QUESTION_MODEL_PATH
    )


def classify_question(question):
    """
    Returns:
        {
            "label": 0/1,
            "type": "direct"/"referential",
            "probabilities": {
                "direct": ...,
                "referential": ...
            }
        }
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Câu hỏi không được rỗng."
        )

    vector = question_vectorizer.transform(
        [question]
    )

    label = int(
        question_classifier.predict(
            vector
        )[0]
    )

    probabilities = {
        "direct": None,
        "referential": None
    }

    if hasattr(
        question_classifier,
        "predict_proba"
    ):
        probs = question_classifier.predict_proba(
            vector
        )[0]

        for cls, prob in zip(
            question_classifier.classes_,
            probs
        ):
            cls = int(cls)

            if cls == DIRECT_LABEL:
                probabilities["direct"] = float(prob)

            elif cls == REFERENTIAL_LABEL:
                probabilities["referential"] = float(prob)

    question_type = (
        "direct"
        if label == DIRECT_LABEL
        else "referential"
    )

    return {
        "label": label,
        "type": question_type,
        "probabilities": probabilities
    }


# ============================================================
# 2. IMAGE RETRIEVAL
# ============================================================

def load_image_retriever():
    global image_model
    global image_preprocess
    global faiss_index
    global image_metadata

    check_file(
        EFFICIENTNET_PATH
    )
    check_file(
        FAISS_INDEX_PATH
    )
    check_file(
        IMAGE_METADATA_PATH
    )

    weights = (
        models
        .EfficientNet_B0_Weights
        .IMAGENET1K_V1
    )

    model = models.efficientnet_b0(
        weights=None
    )

    state_dict = torch.load(
        EFFICIENTNET_PATH,
        map_location=DEVICE
    )

    if isinstance(
        state_dict,
        dict
    ):
        if "model_state_dict" in state_dict:
            state_dict = state_dict[
                "model_state_dict"
            ]

        elif "state_dict" in state_dict:
            state_dict = state_dict[
                "state_dict"
            ]

    model.load_state_dict(
        state_dict,
        strict=True
    )

    # EfficientNet-B0:
    # classifier -> Identity
    # output embedding = 1280 dimensions
    model.classifier = (
        torch.nn.Identity()
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    image_model = model
    image_preprocess = (
        weights.transforms()
    )

    faiss_index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with IMAGE_METADATA_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        image_metadata = json.load(f)

    if faiss_index.ntotal != len(
        image_metadata
    ):
        raise RuntimeError(
            "Số vector trong FAISS không khớp "
            "với số record trong image_metadata.json: "
            f"{faiss_index.ntotal} != {len(image_metadata)}"
        )


@torch.no_grad()
def image_to_embedding(image):
    """
    image:
        PIL.Image.Image
        hoặc đường dẫn ảnh.

    Returns:
        np.ndarray shape (1, 1280), L2-normalized.
    """

    if not isinstance(
        image,
        Image.Image
    ):
        image = Image.open(
            image
        ).convert("RGB")
    else:
        image = image.convert(
            "RGB"
        )

    tensor = image_preprocess(
        image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )

    embedding = image_model(
        tensor
    )

    embedding = (
        embedding
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    embedding = np.ascontiguousarray(
        embedding
    )

    # FAISS index đã build trên vector L2-normalized.
    faiss.normalize_L2(
        embedding
    )

    return embedding


def retrieve_images(
    image,
    top_k=DEFAULT_TOP_K_IMAGE,
    threshold=DEFAULT_IMAGE_THRESHOLD
):
    """
    Tìm Top-K ảnh gần nhất.

    threshold:
        ngưỡng cosine similarity.

    Chỉ trả các ảnh:
        score >= threshold.
    """

    if top_k < 1:
        raise ValueError(
            "top_k phải >= 1."
        )

    embedding = image_to_embedding(
        image
    )

    search_k = min(
        int(top_k),
        faiss_index.ntotal
    )

    scores, indices = (
        faiss_index.search(
            embedding,
            search_k
        )
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0]
    ):
        idx = int(idx)
        score = float(score)

        if idx < 0:
            continue

        if score < threshold:
            continue

        meta = image_metadata[
            idx
        ]

        results.append({
            "rank": len(results) + 1,
            "faiss_index": idx,
            "score": score,
            "image": meta.get(
                "image",
                ""
            ),
            "relic_id": meta.get(
                "relic_id",
                ""
            ),
            "relic_name": meta.get(
                "relic_name",
                ""
            )
        })

    return results


def get_candidate_relics(
    image_results
):
    """
    Mỗi ảnh retrieval suy ra relic_id.

    Loại relic_id trùng nhau nhưng giữ thứ tự
    xuất hiện từ kết quả image retrieval.

    Không vote.
    Không chọn một relic duy nhất.
    """

    candidates = []
    seen = set()

    for item in image_results:
        relic_id = item.get(
            "relic_id"
        )

        if not relic_id:
            continue

        if relic_id in seen:
            continue

        seen.add(
            relic_id
        )

        candidates.append({
            "relic_id": relic_id,
            "relic_name": item.get(
                "relic_name",
                ""
            ),
            # score ảnh tốt nhất của relic này
            # chính là lần xuất hiện đầu tiên vì
            # FAISS đã trả theo score giảm dần.
            "image_score": item[
                "score"
            ]
        })

    return candidates


# ============================================================
# 3. TEXT RETRIEVAL
# ============================================================

def load_text_retriever():
    global tfidf_vectorizer
    global corpus_vectors
    global corpus

    check_file(
        TFIDF_VECTORIZER_PATH
    )
    check_file(
        CORPUS_VECTORS_PATH
    )
    check_file(
        CORPUS_PATH
    )

    tfidf_vectorizer = joblib.load(
        TFIDF_VECTORIZER_PATH
    )

    corpus_vectors = sparse.load_npz(
        CORPUS_VECTORS_PATH
    )

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        corpus = json.load(f)

    if corpus_vectors.shape[0] != len(
        corpus
    ):
        raise RuntimeError(
            "Số vector TF-IDF không khớp "
            "với số chunk trong corpus.json: "
            f"{corpus_vectors.shape[0]} != {len(corpus)}"
        )


def retrieve_chunks(
    question,
    relic_ids=None,
    top_k=DEFAULT_TOP_K_CHUNK,
    threshold=DEFAULT_TEXT_THRESHOLD
):
    """
    relic_ids=None:
        tìm toàn corpus.

    relic_ids=[A, B, C]:
        chỉ tìm trong tất cả chunk thuộc A, B, C.

    Sau đó:
        cosine similarity
        -> threshold
        -> Top-K.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Câu hỏi không được rỗng."
        )

    if top_k < 1:
        raise ValueError(
            "top_k phải >= 1."
        )

    query_vector = (
        tfidf_vectorizer
        .transform(
            [question]
        )
    )

    if relic_ids is None:

        candidate_indices = np.arange(
            len(corpus),
            dtype=np.int64
        )

    else:

        relic_ids = set(
            relic_ids
        )

        candidate_indices = np.array(
            [
                index
                for index, item
                in enumerate(corpus)
                if item["relic_id"]
                in relic_ids
            ],
            dtype=np.int64
        )

    if len(
        candidate_indices
    ) == 0:
        return []

    candidate_vectors = (
        corpus_vectors[
            candidate_indices
        ]
    )

    similarities = cosine_similarity(
        query_vector,
        candidate_vectors
    )[0]

    sorted_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for local_index in sorted_indices:

        score = float(
            similarities[
                local_index
            ]
        )

        if score < threshold:
            # Vì đã sort giảm dần nên các phần tử
            # phía sau cũng không đạt threshold.
            break

        global_index = int(
            candidate_indices[
                local_index
            ]
        )

        item = corpus[
            global_index
        ]

        results.append({
            "rank": len(results) + 1,
            "score": score,
            "chunk_id": item[
                "chunk_id"
            ],
            "relic_id": item[
                "relic_id"
            ],
            "text": item[
                "text"
            ]
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 4. VIT5
# ============================================================

def load_generator():
    global vit5_tokenizer
    global vit5_model

    vit5_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            VIT5_MODEL_NAME
        )
    )

    vit5_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            VIT5_MODEL_NAME
        )
    )

    vit5_model = vit5_model.to(
        DEVICE
    )

    vit5_model.eval()


@torch.no_grad()
def generate_answer(
    question,
    context,
    max_new_tokens=128,
    num_beams=1
):
    """
    Sinh câu trả lời từ question + context.
    """

    if not context:
        raise ValueError(
            "Context không được rỗng."
        )

    input_text = (
        f"Ngữ cảnh: {context}. "
        f"Câu hỏi: {question}. "
        f"Câu trả lời:"
    )

    inputs = vit5_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(
            DEVICE
        )
        for key, value
        in inputs.items()
    }

    output_ids = (
        vit5_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
    )

    answer = (
        vit5_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        .strip()
    )

    return answer


# ============================================================
# 5. LOAD WHOLE SYSTEM
# ============================================================

def load_system():
    global SYSTEM_LOADED

    if SYSTEM_LOADED:
        print(
            "Hệ thống đã được load."
        )
        return

    print(
        f"Device: {DEVICE}"
    )

    print(
        "[1/4] Question classifier..."
    )
    load_question_classifier()
    print("      OK")

    print(
        "[2/4] Image retriever..."
    )
    load_image_retriever()
    print(
        f"      OK - "
        f"{faiss_index.ntotal} images"
    )

    print(
        "[3/4] Text retriever..."
    )
    load_text_retriever()
    print(
        f"      OK - "
        f"{len(corpus)} chunks"
    )

    print(
        "[4/4] ViT5..."
    )
    load_generator()
    print(
        f"      OK - {VIT5_MODEL_NAME}"
    )

    SYSTEM_LOADED = True

    print(
        "\nHệ thống đã sẵn sàng."
    )


# ============================================================
# 6. FULL PIPELINE
# ============================================================

def answer(
    question,
    image=None,
    top_k_image=DEFAULT_TOP_K_IMAGE,
    top_k_chunk=DEFAULT_TOP_K_CHUNK,
    image_threshold=DEFAULT_IMAGE_THRESHOLD,
    text_threshold=DEFAULT_TEXT_THRESHOLD
):
    """
    Chạy pipeline hoàn chỉnh.

    image:
        - None cho direct
        - path hoặc PIL.Image cho referential

    Returns:
        dict chứa toàn bộ intermediate results để debug.
    """

    if not SYSTEM_LOADED:
        raise RuntimeError(
            "Hệ thống chưa được load. "
            "Hãy chạy load_system() trước."
        )

    classification = classify_question(
        question
    )

    question_type = classification[
        "type"
    ]

    image_results = []
    candidate_relics = []

    # ========================================================
    # DIRECT
    # ========================================================

    if question_type == "direct":

        chunk_results = retrieve_chunks(
            question=question,
            relic_ids=None,
            top_k=top_k_chunk,
            threshold=text_threshold
        )

    # ========================================================
    # REFERENTIAL
    # ========================================================

    else:

        if image is None:
            raise ValueError(
                "Câu hỏi được phân loại là referential "
                "nhưng image=None."
            )

        image_results = retrieve_images(
            image=image,
            top_k=top_k_image,
            threshold=image_threshold
        )

        candidate_relics = (
            get_candidate_relics(
                image_results
            )
        )

        candidate_relic_ids = [
            item["relic_id"]
            for item
            in candidate_relics
        ]

        if not candidate_relic_ids:

            return {
                "question": question,
                "classification": classification,
                "image_results": image_results,
                "candidate_relics": [],
                "chunk_results": [],
                "context": None,
                "answer": None,
                "status": "no_image_candidate"
            }

        chunk_results = retrieve_chunks(
            question=question,
            relic_ids=candidate_relic_ids,
            top_k=top_k_chunk,
            threshold=text_threshold
        )

    # ========================================================
    # NO TEXT RESULT
    # ========================================================

    if not chunk_results:

        return {
            "question": question,
            "classification": classification,
            "image_results": image_results,
            "candidate_relics": candidate_relics,
            "chunk_results": [],
            "context": None,
            "answer": None,
            "status": "no_text_candidate"
        }

    # ========================================================
    # GENERATION
    # ========================================================

    # Hiện tại ViT5 nhận TOP-1 chunk làm context.
    context = chunk_results[0][
        "text"
    ]

    generated_answer = generate_answer(
        question=question,
        context=context
    )

    return {
        "question": question,
        "classification": classification,
        "image_results": image_results,
        "candidate_relics": candidate_relics,
        "chunk_results": chunk_results,
        "context": context,
        "answer": generated_answer,
        "status": "ok"
    }


# ============================================================
# 7. DEBUG PRINT
# ============================================================

def print_result(result):
    """
    In toàn bộ pipeline để test.
    Không dùng cho giao diện cuối.
    """

    print(
        "=" * 70
    )

    print(
        "QUESTION"
    )

    print(
        "=" * 70
    )

    print(
        result["question"]
    )

    print(
        "\nClassification:"
    )

    print(
        json.dumps(
            result["classification"],
            ensure_ascii=False,
            indent=2
        )
    )

    if result[
        "image_results"
    ]:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "IMAGE RESULTS"
        )

        print(
            "=" * 70
        )

        for item in result[
            "image_results"
        ]:

            print(
                f"#{item['rank']} "
                f"score={item['score']:.4f} | "
                f"{item['relic_name']} | "
                f"{item['relic_id']}"
            )

        print(
            "\nCandidate relics:"
        )

        for item in result[
            "candidate_relics"
        ]:

            print(
                f"- {item['relic_name']} "
                f"({item['relic_id']}) "
                f"| image_score="
                f"{item['image_score']:.4f}"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEXT RESULTS"
    )

    print(
        "=" * 70
    )

    for item in result[
        "chunk_results"
    ]:

        print(
            f"\n#{item['rank']} "
            f"score={item['score']:.4f}"
        )

        print(
            f"relic_id: "
            f"{item['relic_id']}"
        )

        print(
            f"chunk_id: "
            f"{item['chunk_id']}"
        )

        print(
            item["text"]
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONTEXT USED BY VIT5"
    )

    print(
        "=" * 70
    )

    print(
        result["context"]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VIT5 ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        result["answer"]
    )

    print(
        "\nStatus:",
        result["status"]
    )


# ============================================================
# 8. COLAB UI
# ============================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_IMAGE_PATH = None


def get_uploaded_file_info():
    """Đọc file hiện tại từ FileUpload, tương thích ipywidgets 7/8."""
    value = image_upload.value

    if not value:
        return None, None

    if isinstance(value, (tuple, list)):
        item = value[0]

        if isinstance(item, dict):
            filename = (
                item.get("name")
                or item.get("metadata", {}).get("name")
                or "uploaded_image.jpg"
            )
            content = item.get("content")
        else:
            filename = getattr(item, "name", "uploaded_image.jpg")
            content = getattr(item, "content", None)

    elif isinstance(value, dict):
        first_key = next(iter(value.keys()))
        item = value[first_key]
        filename = first_key

        if isinstance(item, dict):
            filename = item.get("name", filename)
            content = item.get("content")
        else:
            filename = getattr(item, "name", filename)
            content = getattr(item, "content", None)

    else:
        return None, None

    if content is None:
        raise ValueError("Không đọc được dữ liệu ảnh upload.")

    return filename, bytes(content)


def delete_current_image():
    """Xóa ảnh active hiện tại khỏi Colab."""
    global CURRENT_IMAGE_PATH

    if CURRENT_IMAGE_PATH is not None:
        path = Path(CURRENT_IMAGE_PATH)
        if path.exists():
            path.unlink()

    CURRENT_IMAGE_PATH = None


def save_uploaded_image():
    """
    Lưu ảnh upload thành uploads/current_image.*.
    Mỗi thời điểm chỉ giữ đúng 1 ảnh active.
    """
    global CURRENT_IMAGE_PATH

    filename, content = get_uploaded_file_info()

    if filename is None:
        return None

    delete_current_image()

    extension = Path(filename).suffix.lower()
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if extension not in valid_extensions:
        extension = ".jpg"

    save_path = UPLOAD_DIR / f"current_image{extension}"

    with save_path.open("wb") as f:
        f.write(content)

    CURRENT_IMAGE_PATH = str(save_path)
    return CURRENT_IMAGE_PATH


def show_current_image():
    """Preview ảnh active hiện tại."""
    with image_output:
        clear_output(wait=True)

        if CURRENT_IMAGE_PATH is None:
            return

        path = Path(CURRENT_IMAGE_PATH)
        if not path.exists():
            return

        image = Image.open(path).convert("RGB")
        preview = image.copy()
        preview.thumbnail((450, 350))
        display(preview)


def on_image_selected(change):
    try:
        path = save_uploaded_image()
        if path is not None:
            show_current_image()
    except Exception as e:
        with image_output:
            clear_output(wait=True)
            print(f"Lỗi upload ảnh: {e}")


def on_send_clicked(button):
    with answer_output:
        clear_output(wait=True)

        try:
            question = question_box.value.strip()

            if not question:
                print("Vui lòng nhập câu hỏi.")
                return

            result = answer(
                question=question,
                image=CURRENT_IMAGE_PATH,
                top_k_image=TOP_K_IMAGE_WIDGET.value,
                top_k_chunk=TOP_K_CHUNK_WIDGET.value,
                image_threshold=IMAGE_THRESHOLD_WIDGET.value,
                text_threshold=TEXT_THRESHOLD_WIDGET.value,
            )

            if result["answer"] is not None:
                print(result["answer"])
            elif result["status"] == "no_image_candidate":
                print("Không có ảnh tương đồng đạt ngưỡng.")
            elif result["status"] == "no_text_candidate":
                print("Không tìm thấy ngữ cảnh phù hợp.")
            else:
                print("Không thể sinh câu trả lời.")

        except Exception as e:
            print(f"Lỗi: {e}")


def on_clear_clicked(button):
    question_box.value = ""
    delete_current_image()

    try:
        image_upload.value = ()
    except Exception:
        try:
            image_upload.value.clear()
        except Exception:
            pass

    with image_output:
        clear_output()

    with answer_output:
        clear_output()


def build_ui():
    """Tạo và hiển thị giao diện demo trên Colab."""
    global question_box
    global image_upload
    global send_button
    global clear_button
    global image_output
    global answer_output
    global TOP_K_IMAGE_WIDGET
    global TOP_K_CHUNK_WIDGET
    global IMAGE_THRESHOLD_WIDGET
    global TEXT_THRESHOLD_WIDGET

    title = widgets.HTML("<h3>Trợ lý ảo di tích lịch sử</h3>")

    question_box = widgets.Textarea(
        value="",
        placeholder="Nhập câu hỏi...",
        description="Câu hỏi:",
        layout=widgets.Layout(width="700px", height="100px"),
        style={"description_width": "70px"},
    )

    image_upload = widgets.FileUpload(
        accept="image/*",
        multiple=False,
        description="Chọn ảnh",
    )

    TOP_K_IMAGE_WIDGET = widgets.BoundedIntText(
        value=DEFAULT_TOP_K_IMAGE,
        min=1,
        max=100,
        step=1,
        description="Top-K ảnh:",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "100px"},
    )

    IMAGE_THRESHOLD_WIDGET = widgets.BoundedFloatText(
        value=DEFAULT_IMAGE_THRESHOLD,
        min=-1.0,
        max=1.0,
        step=0.01,
        description="Ngưỡng ảnh:",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "100px"},
    )

    TOP_K_CHUNK_WIDGET = widgets.BoundedIntText(
        value=DEFAULT_TOP_K_CHUNK,
        min=1,
        max=50,
        step=1,
        description="Top-K chunk:",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "100px"},
    )

    TEXT_THRESHOLD_WIDGET = widgets.BoundedFloatText(
        value=DEFAULT_TEXT_THRESHOLD,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Ngưỡng text:",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "100px"},
    )

    send_button = widgets.Button(
        description="Gửi",
        button_style="primary",
        icon="search",
    )

    clear_button = widgets.Button(
        description="Xóa",
        icon="trash",
    )

    image_output = widgets.Output(
        layout=widgets.Layout(width="700px")
    )

    answer_output = widgets.Output(
        layout=widgets.Layout(
            width="700px",
            border="1px solid #dddddd",
            padding="12px",
        )
    )

    image_upload.observe(on_image_selected, names="value")
    send_button.on_click(on_send_clicked)
    clear_button.on_click(on_clear_clicked)

    retrieval_row_1 = widgets.HBox([
        TOP_K_IMAGE_WIDGET,
        IMAGE_THRESHOLD_WIDGET,
    ])

    retrieval_row_2 = widgets.HBox([
        TOP_K_CHUNK_WIDGET,
        TEXT_THRESHOLD_WIDGET,
    ])

    buttons = widgets.HBox([
        image_upload,
        send_button,
        clear_button,
    ])

    ui = widgets.VBox([
        title,
        question_box,
        retrieval_row_1,
        retrieval_row_2,
        buttons,
        widgets.HTML("<b>Ảnh hiện tại:</b>"),
        image_output,
        widgets.HTML("<b>Câu trả lời:</b>"),
        answer_output,
    ])
    return ui 
    display(ui)


# ============================================================
# 9. START
# ============================================================


load_system()
ui = build_ui()
