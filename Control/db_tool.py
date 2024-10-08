from pymysql import Connection


class Mysql():
    def __init__(self):
        self.con = None
        # create DB connection
        try:
            self.con = Connection(
                host='localhost',
                port=3306,
                user='root',
                passwd='123mysql',
                database='msa_vis',
            )
            # create cursor
            self.cur = self.con.cursor()

        except Exception as e:
            print('Connection failed:', e)

    def insert_table(self, user_text, machine_text):
        sql = """
            INSERT INTO diolague (user_text, machine_text) VALUES (%s, %s)
        """
        data = (user_text, machine_text)
        self.cur.execute(sql, data)
        self.con.commit()

    def check_table(self):
        his_text = ''
        sql = """
        SELECT * FROM (
            SELECT * FROM diolague ORDER BY chat_id DESC LIMIT 20
        ) AS subquery
        ORDER BY chat_id ASC;
        """
        self.cur.execute(sql)
        res = self.cur.fetchall()
        for row in res:
            his_text = his_text + f'[user: {row[1]}, machine: {row[2]}]'

        return his_text

    def clear_table(self):
        sql = "TRUNCATE TABLE diolague"
        self.cur.execute(sql)  # 执行sql代码
        self.con.commit()  # 更新表

    def close_connection(self):
        self.cur.close()


if __name__ == '__main__':
    db = Mysql()
    db.check_table()
