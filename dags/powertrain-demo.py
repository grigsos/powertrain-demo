from airflow import DAG
from conveyor.operators import ConveyorContainerOperatorV2
from datetime import datetime, timedelta


default_args = {
    "owner": "Conveyor",
    "depends_on_past": False,
    "start_date": datetime(year=2024, month=8, day=22),
    "email": [],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "powertrain-demo", default_args=default_args, schedule_interval="@daily", max_active_runs=1
)

ConveyorContainerOperatorV2(
    dag=dag,
    task_id="sample",
    cmds=["python"],
    arguments=[
        "-m",
        "powertraindemo.sample"
    ],
    instance_type="mx.micro",
    aws_role="powertrain-demo-{{ macros.conveyor.env() }}",
)
