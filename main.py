import cv2
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import asyncio
from face_utils import FaceDetector, FaceEmbedder
from face_utils import add_face, recognize_face

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App (RetinaFace + Async DB)")
        self.root.resizable(False, False)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            messagebox.showerror("Lỗi không mở được webcam")
            root.destroy()
            return
        
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, pady = 10)

        self.btn_add = tk.Button(btn_frame, text="Thêm khuôn mặt", width=15, command=self.on_add_face)
        self.btn_add.pack(side=tk.LEFT, padx=5)

        self.btn_recog = tk.Button(btn_frame, text="Nhận diện", width=15, command=self.on_recognize_face)
        self.btn_recog.pack(side=tk.LEFT, padx=5)

        self.btn_quit = tk.Button(btn_frame, text="Thoát", width=15, command=self.quit_app)
        self.btn_quit.pack(side=tk.RIGHT, padx=5)

        # 5. Label hiển thị kết quả
        self.lbl_result = tk.Label(root, text="Kết quả sẽ hiển thị ở đây.", font=("Arial", 12))
        self.lbl_result.pack(pady=5)

        # 6. Biến lưu frame hiện tại
        self.current_frame = None

        # 7. Bắt đầu vòng lặp hiển thị
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.imgtk = imgtk

        self.root.after(15, self.update_frame)

    def on_add_face(self):
        if self.current_frame is None:
            return

        frame = self.current_frame
        bboxes = self.detector.detect(frame)
        if len(bboxes) == 0:
            messagebox.showinfo("Kết quả", "Không tìm thấy khuôn mặt nào để thêm.")
            return
        
        x1, y1, x2, y2 = bboxes[0]
        face_crop = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))

        emb = self.embedder.get_embedding(face_rgb)
        norm = np.linalg.norm(emb)

        if norm > 0:
            emb = emb / norm

        name = simpledialog.askstring("Nhập tên", "Nhập tên cho khuôn mặt này:")
        if not name:
            self.lbl_result.config(text="Đã huỷ thêm khuôn mặt.")
            return

        try:
            asyncio.run(add_face(name, emb))
            self.lbl_result.config(text=f"Đã lưu khuôn mặt: {name}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không lưu được: {e}")

    def on_recognize_face(self):
        if self.current_frame is None:
            return

        frame = self.current_frame
        bboxes = self.detector.detect(frame)
        if not bboxes:
            messagebox.showinfo("Kết quả", "Không tìm thấy khuôn mặt nào để nhận diện.")
            return

        x1, y1, x2, y2 = bboxes[0]
        face_crop = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))

        emb = self.embedder.get_embedding(face_rgb)
        norm = np.linalg.norm(emb)

        if norm > 0:
            emb = emb / norm

        if len(bboxes) == 0:
                self.root.after(0, lambda: self._show_result("❌ Lỗi tạo embedding!", 'red'))
                return

        try:
            name, similarity = asyncio.run(recognize_face(emb, threshold=0.7))
            if name:
                self.lbl_result.config(text=f"Nhận diện: {name} (similarity={similarity:.3f})")
            else:
                self.lbl_result.config(text="Không khớp với bất kỳ khuôn mặt nào.")
        except Exception as e:
            print(e)
            messagebox.showerror("Lỗi", f"Nhận diện thất bại: {e}")
        
    def quit_app(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
