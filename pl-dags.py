from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
# [END default_args]

# [START instantiate_dag]
with DAG(
    dag_id='a-pipe-test',
    default_args=default_args,
    description='A entire pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['ver1'],
) as dag:

    # [END basic_task]
    dag.doc_md = """
    开始测试的一整套的流程
    """  # otherwise, type it like this
    # [END documentation]

    p_task_1_1 = BashOperator(
        task_id="pull_code",      # s3复制数据集 code commit拉取代码
        bash_command='cd /home/ec2-user/work;git pull;'
        )

    p_task_2_2 = BashOperator(
        task_id="train_model",      # 拉取模型
        bash_command="/usr/bin/python3 /home/ec2-user/work/pipel.py"
    )


    p_task_1_1 >> p_task_2_2