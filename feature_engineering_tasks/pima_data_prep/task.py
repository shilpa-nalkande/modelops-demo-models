from teradataml import *
from aoa import (aoa_create_context, ModelContext)
import pandas as pd
import numpy as np
from sqlalchemy import literal_column


def run_task(context: ModelContext, **kwargs):
    # aoa_create_context()
    df = DataFrame.from_query("SELECT * FROM DEMO_ModelOps.pima_patient_features")
    

    from sqlalchemy import literal_column
    from teradatasqlalchemy import DATE
    import numpy as np
    from sqlalchemy import func
    current_date_ = func.current_date()
    column = literal_column("CAST('1950-01-01' AS DATE) + (CAST('2000-01-01' AS DATE) - CAST('1950-01-01' AS DATE)) * RANDOM(1,100)", type_=DATE)
    df = df.assign(current_dt_col = current_date_)
    df = df.assign(birthday = (df.current_dt_col - df.Age * 365))
    # return(df)
    df = df.assign(calculated_age = (df.current_dt_col - df.birthday)/365)
    
#     print(df)
#     # Calculate age
#     df_pd['calculated_age'] = df_pd['birthday'].apply(lambda x: (pd.to_datetime('today') - x).days // 365)
    
#     # Remove the original age column
#     df_pd = df_pd.drop(columns=['Age'])
    # return(df)
    scale_fit = ScaleFit(
        data=df,
        target_columns=["NumTimesPrg","PlGlcConc","BloodP","SkinThick","TwoHourSerIns","BMI","DiPedFunc",
                        "Age","calculated_age"],
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
    
#     Feature engineering
    fs = FeatureStore(repo='demo_user')
    fs.setup()
    fg = FeatureGroup.from_DataFrame(name='PIMA', df=tdf, entity_columns='PatientId')
    fs.apply(fg)
    df = fs.get_dataset('PIMA')
    # df
    # print(df)
    with open(f"{context.artifact_output_path}/transformation_report.txt", "w") as f:
        print(tdf.describe(), file=f)
    
    # Store build properties as a file artifact
    with open(f"{context.artifact_output_path}/build_properties.txt", "w") as f:
        f.write(str(kwargs))
        
    # return(df)
