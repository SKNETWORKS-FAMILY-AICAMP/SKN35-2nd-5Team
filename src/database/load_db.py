from dotenv import load_dotenv

from .db import get_db_connection

load_dotenv()

COLUMN_MAPPING = {
    "Employee ID": "employee_id",
    "Age": "age",
    "Gender": "gender",
    "Years at Company": "years_at_company",
    "Job Role": "job_role",
    "Monthly Income": "monthly_income",
    "Work-Life Balance": "work_life_balance",
    "Job Satisfaction": "job_satisfaction",
    "Performance Rating": "performance_rating",
    "Number of Promotions": "number_of_promotions",
    "Overtime": "overtime",
    "Distance from Home": "distance_from_home",
    "Education Level": "education_level",
    "Marital Status": "marital_status",
    "Number of Dependents": "number_of_dependents",
    "Job Level": "job_level",
    "Company Size": "company_size",
    "Company Tenure": "company_tenure",
    "Remote Work": "remote_work",
    "Leadership Opportunities": "leadership_opportunities",
    "Innovation Opportunities": "innovation_opportunities",
    "Company Reputation": "company_reputation",
    "Employee Recognition": "employee_recognition",
    "Attrition": "attrition",
    "Industry Experience Gap": "industry_experience_gap",
    "Promotion Rate": "promotion_rate",
}


def get_raw_data(data_type):
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT *
            FROM employee_attrition_raw
            WHERE type = %s
        """

        cursor.execute(query, (data_type,))

        rows = cursor.fetchall()

        return rows

    except Exception:
        if connection:
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()


def get_processed_data(data_type):
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT *
            FROM employee_attrition_processed
            WHERE type = %s
        """

        cursor.execute(query, (data_type,))

        rows = cursor.fetchall()

        return rows

    except Exception:
        if connection:
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()


# if __name__ == "__main__":
# get_raw_data("train")
# get_raw_data("test")

# get_processed_data("train")
# get_processed_data("test")
