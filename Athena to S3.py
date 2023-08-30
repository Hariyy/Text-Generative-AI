import boto3
import pandas as pd
import io
import re
import time

def athena_query(client, params):
    
    response = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={
            'Database': params['database']
        },
        ResultConfiguration={
            'OutputLocation': 's3://' + params['bucket'] + '/' + params['path']
        }
    )
    return response

def athena_to_s3(session, params, max_execution =1000):
    client = session.client('athena', region_name=params["region"])
    execution = athena_query(client, params)
    execution_id = execution['QueryExecutionId']
    state = 'RUNNING'

    while (max_execution > 0 and state in ['RUNNING', 'QUEUED']):
        max_execution = max_execution - 1
        response = client.get_query_execution(QueryExecutionId = execution_id)

        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            if state == 'FAILED':
                return False
            elif state == 'SUCCEEDED':
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]                                            
                
                return filename
        time.sleep(1)
    
    return False


# Deletes all files in your path so use carefully!
def cleanup(session, params):
    s3 = session.resource('s3')
    my_bucket = s3.Bucket(params['bucket'])
    for item in my_bucket.objects.filter(Prefix=params['path']):
        item.delete()

AnnualEnroll = '''
         SELECT 
                *
               
         FROM 
                adl_core_prod_marketscan_db.annualenroll 
         where 
                 partition_0 = 'commercial' 
        and     enrolid IS  NOT NULL
        and     year = 2019
        and     emprel  = 1
        and     msa is not null
        and     msa <> 0
      
        
         '''


AnnualEnrollparams = {
    'region': 'us-east-1',
    'database': 'adl_core_prod_marketscan_db',
    'bucket': 'adl-core-ml-staging',
    'path': 'Hype/aip-data-engineering-Athena-output/MarketScan/AnnualEnroll',
    'query': AnnualEnroll
}


session = boto3.Session()

# Removes all files from the s3 folder you specified, so be careful
#cleanup(session, AnnualEnrollparams)





# Pull all the required data from Athena and write to S3
# AnnualEnroll_s3_filename = athena_to_s3(session, AnnualEnrollparams)




