import psycopg2

def init_db():
    connection = psycopg2.connect(
        dbname="postgres",
        user="postgres.mbtjtlvdxoznmbjogpmw",
        password="!NLP_MISIS1012",
        host="aws-0-us-west-1.pooler.supabase.com",
        port="5432"
    )
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        language VARCHAR(10) NOT NULL,
        content TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS elements (
        id SERIAL PRIMARY KEY,
        article_id INT REFERENCES articles(id),
        type VARCHAR(20) NOT NULL,
        path TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS fragments (
        id SERIAL PRIMARY KEY,
        article_id INT REFERENCES articles(id),
        element_id INT,
        content TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS preprocessing_results (
        id SERIAL PRIMARY KEY,
        fragment_id INT REFERENCES fragments(id),
        step VARCHAR(50) NOT NULL,
        processed_text TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS graphs (
        id SERIAL PRIMARY KEY,
        fragment_id INT REFERENCES fragments(id),
        name VARCHAR(100) NOT NULL,
        type VARCHAR(50) NOT NULL,
        graph_data JSON NOT NULL
    );
    """)

    connection.commit()
    cursor.close()
    connection.close()
    print("База данных успешно инициализирована")

if __name__ == "__main__":
    init_db()