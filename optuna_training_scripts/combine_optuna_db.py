import sqlite3
import pandas as pd

def combine_optuna_dbs(db1_path, db2_path):
    """"
    This function combines two optuna databases into the db1.
    Both dbs should be from the same study.
    The ids of the db2 are updated to be unique.
    """

    # get all the tables names
    with sqlite3.connect(db1_path) as con:
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        tables = [table[0] for table in tables]
    not_tables = ['studies', 'version_info', 'study_directions', 'alembic_version']
    tables = [table for table in tables if table not in not_tables]

    dfs1 = {}
    with sqlite3.connect(db1_path) as con:
        for table in tables:
            dfs1[table] = pd.read_sql_query(f"SELECT * FROM {table}", con)

    table_ids = {'study_user_attributes' : ['study_user_attribute_id'],
                'study_system_attributes' : ['study_system_attribute_id'],
                'trials' : ['trial_id', 'number'],
                'trial_user_attributes' : ['trial_user_attribute_id'],
                'trial_system_attributes' : ['trial_system_attribute_id'],
                'trial_params' : ['param_id'],
                'trial_values' : ['trial_value_id'],
                'trial_intermediate_values' : ['trial_intermediate_value_id'],
                'trial_heartbeats' : ['trial_heartbeat_id'],
    }

    max_ids = {}
    # add max id if the table is not empty
    for table in tables:
        if len(dfs1[table]) > 0:
            for id in table_ids[table]:
                max_ids[id] = dfs1[table].iloc[-1][id]
        else:
            for id in table_ids[table]:
                max_ids[id] = 0

    dfs2 = {}
    with sqlite3.connect(db2_path) as con:
        for table in tables:
            dfs2[table] = pd.read_sql_query(f"SELECT * FROM {table}", con)

    update_ids = {'study_user_attributes' : ['study_user_attribute_id'],
             'study_system_attributes' : ['study_system_attribute_id'],
            'trials' : ['trial_id', 'number'],
            'trial_user_attributes' : ['trial_user_attribute_id', 'trial_id'],
            'trial_system_attributes' : ['trial_system_attribute_id', 'trial_id'],
            'trial_params' : ['param_id', 'trial_id'],
            'trial_values' : ['trial_value_id', 'trial_id'],
            'trial_intermediate_values' : ['trial_intermediate_value_id', 'trial_id'],
            'trial_heartbeats' : ['trial_heartbeat_id', 'trial_id'],
    }

    # update the ids of the second db
    for table in tables:
        if len(dfs2[table]) > 0:
            for id in update_ids[table]:
                dfs2[table][id] = dfs2[table][id] + max_ids[id]

    # add the second db to the first db
    with sqlite3.connect(db1_path) as con1:
        for table in tables:
            dfs2[table].to_sql(table, con1, if_exists='append', index=False)

combine_optuna_dbs("/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/optuna_training_scripts/htdemucs_optimization_2025-03-04_cc7ee30f copy.sqlite3", "/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/optunadb/htdemucs_optimization_2025-03-02_ed1be10e.sqlite3")