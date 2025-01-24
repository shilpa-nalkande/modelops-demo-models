from teradataml import *
from aoa import (aoa_create_context, ModelContext)
import pandas as pd
import numpy as np
from sqlalchemy import literal_column


def run_task(context: ModelContext, **kwargs):
    aoa_create_context()
    df = DataFrame.from_query("SELECT * FROM DEMO_ModelOps.pima_patient_features")
    
    start_date = CAST('1950-01-01' AS DATE)
    end_date = CAST('2000-01-01' AS DATE)
    
    column = literal_column(start_date + (end_date - start_date) * np.random.rand(len(df_pd)))
    df = df.assign('birthday' = column)
    df = df.assign('calculated_age' = (CURRENT_DATE - df.birthday)/365)
    
    print(df)
#     # Calculate age
#     df_pd['calculated_age'] = df_pd['birthday'].apply(lambda x: (pd.to_datetime('today') - x).days // 365)
    
#     # Remove the original age column
#     df_pd = df_pd.drop(columns=['Age'])
    
    scale_fit = ScaleFit(
        data=df,
        target_columns=["1:"],
        scale_method="RANGE",
        miss_value="KEEP",
        global_scale=False
    ).output
    
    transformed_data = ColumnTransformer(
        input_data=df,
        scale_fit_data=scale_fit,
    ).result
    
    # Write DataFrame to a Teradata table
    copy_to_sql(
        df=transformed_data,
        table_name='transformed_data',
        if_exists='replace'
    )
    
    # Create a teradataml DataFrame from the table
    tdf = DataFrame('transformed_data')
    
    print(df)
    with open(f"{context.artifact_output_path}/transformation_report.txt", "w") as f:
        print(tdf.describe(), file=f)
    
    # Store build properties as a file artifact
    with open(f"{context.artifact_output_path}/build_properties.txt", "w") as f:
        f.write(str(kwargs))
