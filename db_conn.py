import asyncio
import asyncpg
from pgvector.asyncpg import register_vector


async def get_db_pool():
    pool = await asyncpg.create_pool(
        host = "localhost",
        port = 5432,
        database = "face",
        user = "postgres",
        password = "bach2612",
        min_size = 1,
        max_size = 5
    )
    return pool

async def init_db():
    """
    - Lấy pool
    - Acquire connection từ pool
    - Đăng ký VECTOR adapter, tạo extension nếu chưa có
    - Tạo bảng faces nếu chưa tồn tại
    - Đóng pool
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # 2.1. Đăng ký kiểu vector cho asyncpg
        await register_vector(conn)

        # 2.2. Tạo extension pgvector nếu chưa có
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2.3. Tạo bảng `faces` nếu chưa tồn tại
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                embedding VECTOR(512)
            );
        """)
        print("✅ Extension + bảng 'faces' đã được khởi tạo hoặc đã tồn tại.")

    # 2.4. Đóng pool khi không dùng nữa
    await pool.close()

async def test_async_connection():
    """
    Mở pool, acquire 1 kết nối, 
    thực hiện truy vấn SELECT version() và kiểm tra xem table có tồn tại.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Đăng ký VECTOR (để chắc chắn có adapter)
        await register_vector(conn)

        # Lấy version
        version = await conn.fetchval("SELECT version();")
        print("✅ Kết nối async thành công. PostgreSQL version:", version)

        # Kiểm tra xem table faces có tồn tại
        table = await conn.fetchval("""
            SELECT to_regclass('public.faces');
        """)  # trả về 'faces' nếu table đã tồn tại, None nếu không
        if table:
            print("✅ Table 'faces' đã tồn tại trong database.")
        else:
            print("⚠ Table 'faces' chưa tồn tại.")


    await pool.close()

if __name__ == "__main__":
    asyncio.run(init_db())
    asyncio.run(test_async_connection())