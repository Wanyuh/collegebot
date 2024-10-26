import asyncio
import asyncpg


async def fetch_data():
    # Establish a connection to the database
    connection = await asyncpg.connect(
        user='postgres',
        password='thisisatest',
        database='postgres',
        host='rag-project.cr482uc82ad7.us-west-2.rds.amazonaws.com',
        port='5432'
    )

    # Execute a SQL query
    rows = await connection.fetch("SELECT * FROM max_kb_embedding")

    # Print results
    for row in rows:
        print(row)

    # Close the connection
    await connection.close()


async def fetch_tables():
    # Establish a connection to the database
    connection = await asyncpg.connect(
        user='postgres',
        password='thisisatest',
        database='postgres',  # Use the correct database name
        host='rag-project.cr482uc82ad7.us-west-2.rds.amazonaws.com',
        port='5432'
    )

    # Execute a SQL query to get all table names
    rows = await connection.fetch("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'  
    """)

    # Print table names
    for row in rows:
        print(row['table_name'])

    # Close the connection
    await connection.close()

if __name__ == '__main__':

    # Run the async function
    # asyncio.run(fetch_data())
    asyncio.run(fetch_tables())
