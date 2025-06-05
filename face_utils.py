import numpy as np
import asyncpg
import asyncio
from pgvector.asyncpg import register_vector
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from db_conn import get_db_pool

async def add_face(name: str, embedding: np.ndarray):
    """
    Thêm một khuôn mặt mới vào bảng `faces`.
    - name: chuỗi tên (vd. "AnhTu")
    - embedding: numpy array kích thước (512,), giá trị embedding đã tính từ ArcFace
    """
    # 1. Lấy pool asyncpg (kết nối bất đồng bộ)
    pool = await get_db_pool()

    # 2. Mượn connection từ pool
    async with pool.acquire() as conn:
        # 3. Đăng ký adapter VECTOR cho connection này
        await register_vector(conn)

        # 4. Chuyển embedding (numpy array) thành Python list
        vector_list = embedding.tolist()

        # 5. Thực thi INSERT
        await conn.execute(
            """
            INSERT INTO faces (name, embedding)
            VALUES ($1, $2)
            """,
            name,
            vector_list
        )

    # 6. Đóng pool sau khi xong (trả mọi kết nối về pool, rồi pool sẽ tự close)
    await pool.close()

async def recognize_face(embedding: np.ndarray, threshold: float = 0.7):
    """
    Truy vấn database để nhận diện khuôn mặt dựa vào embedding.
    - embedding: numpy array (512,)
    - threshold: ngưỡng cosine distance để nhận diện (mặc định 0.4)
    Trả về (name, distance) nếu tìm thấy bản ghi phù hợp; ngược lại (None, None).
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Đăng ký adapter VECTOR
        await register_vector(conn)

        vector_list = embedding.tolist()

        # Truy vấn:
        #   - operator "<#>" do pgvector cung cấp để tính cosine distance
        #   - ORDER BY distance LIMIT 1 để lấy bản ghi gần nhất
        row = await conn.fetchrow(
            """
                    SELECT name,
                    (1 - (embedding <=> $1)) AS similarity
                    FROM faces
                    ORDER BY similarity DESC
                    LIMIT 1;
            """,
            vector_list
        )

        if row and row["similarity"] >= threshold:
            # Nếu similarity > threshold, trả về tên và giá trị similarity (float)
            return row["name"], float(row["similarity"])
        else:
            # Không thỏa threshold => không khớp
            return None, None

    await pool.close()



class FaceDetector:
    def __init__(self):
        self.detector = MTCNN(keep_all=True)

    def detect(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.detector.detect(img_rgb)
        if boxes is None:
            return []
        # Chuyển mỗi box thành tuple int và gói vào Python list
        h, w = img_bgr.shape[:2]
        result = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # Giới hạn trong khung ảnh cho chắc:
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            result.append((x1, y1, x2, y2))
        return result
    
class FaceEmbedder:
    def __init__(self):
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def get_embedding(self, face_rgb: np.ndarray):
        """
        - face_rgb: crop RGB (160×160)
        - Trả về embedding numpy (512,)
        """
        img = Image.fromarray(face_rgb)
        img_tensor = self.transform(img).unsqueeze(0)  # (1,3,160,160)
        with torch.no_grad():
            emb = self.embedder(img_tensor)  # (1,512)
        return emb.squeeze(0).cpu().numpy()
