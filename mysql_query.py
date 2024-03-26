import mysql.connector
import datetime

class MysqlQuery:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

    def write_data_into_users(self, data):
        cursor = self.conn.cursor()
        insert_query = "INSERT INTO users (username, email, password, shift_id) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_query, data)
        self.conn.commit()
        print("successfully")

    def write_data_into_attendance(self, data):
        cursor = self.conn.cursor()
        insert_query = "INSERT INTO attendances (employee_id, check_status) VALUES (%s, %s)"
        cursor.execute(insert_query, data)
        self.conn.commit()
        print("successfully")

    def get_user_id_by_username(self, username):
        cursor = self.conn.cursor()
        query = "SELECT id FROM employees WHERE CONCAT(LOWER(first_name), ' ', LOWER(last_name)) = %s"
        cursor.execute(query, (username,))
        user_id = cursor.fetchone()
        if user_id:
            return user_id[0]
        else:
            return None

    def check_in_or_out(self, user_id, period):
        current_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
        cursor = self.conn.cursor()
        query = "SELECT check_status FROM attendances WHERE employee_id = %s AND DATE(time_stamp) = %s AND time(time_stamp) BETWEEN %s AND %s ORDER BY time_stamp DESC LIMIT 1"
        cursor.execute(query, (user_id, current_date, period[0], period[1]))
        check_status = cursor.fetchone()
        if check_status:
            return check_status[0]
        else:
            return None

    def close_connection(self):
        if self.conn.is_connected():
            self.conn.close()
            print("Connection closed.")
        else:
            print("No active connection to close.")

# # Example usage:
# attendance_system = MysqlQuery(host="localhost", user="root", password="", database="attendance_db")
# # You can call your methods on this object now
# # attendance_system.write_data_into_users(('sample_user', 'sample_email@example.com', 'sample_password', 1))
# # user_id = attendance_system.get_user_id_by_username("sample_user")
# last_status = attendance_system.check_in_or_out(user_id=12, period=['8:00:00', '16:00:00'])
# print(last_status)
# attendance_system.close_connection()
