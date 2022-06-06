from flask import Flask, render_template, flash,  url_for, session, request,redirect,send_file
from functools import wraps
import pandas as pd
import datetime as dt
import time
import uuid
import multiprocessing as mp
import os
import sys
sys.path.append('app_files')
sys.path.append(os.path.join('app_files','sparcc_master'))
from mathematical_funcs import Signaling_Metabolite_v9, unique, filter_data_aundance_v6_OneDim, file_exceler_SigMeta
from mathematical_funcs2 import Bacteria_Influence_v1



# Configurations **********************************************************

app = Flask(__name__)
app.config['SECRET_KEY'] = 'DontTellAnyOne'
app.config['UPLOAD_FOLDER'] = 'Upload_folder'
app.config['OUTPUT_FOLDER'] = 'Output_folder'

MULTIPROCESS_number = 1 # Number active multiprecesses at the same time (it is chosen
                        # to be 1 as the Challenge2 asked. On the other words, limited_f
                        # function is going to run just 1 time in background)
global MULTIPROCESS_list
MULTIPROCESS_list = []  # a list to store multiprecessing objects

ALLOWED_EXTENSIONS = {'csv'}
status_manager_ADDRESS = os.path.join('app_information','status_manager.csv')
process_list_ADDRESS = os.path.join('app_information','process_list.csv')
# end Configurations *******************************************************


# #Login requirement=================================
# # Every user must login before working with the app
# def is_logged_in(f):
#     @wraps(f)
#     def wrap(*args, **kwargs):
#         if 'Logged_in' in session:
#             return f(*args, **kwargs)
#         else:
#             return redirect(url_for('loginP'))
#     return wrap
# # ==================================================

# #Logout page==========================
# #Logout address that every user can logout with
# @app.route('/index', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')
# #end Logout page==========================

# Web Pages **********************************************************
# #Login page==========================
# @app.route('/login', methods=['GET', 'POST'])
# def loginP():
#     if request.method == 'POST':
#         #GET form
#         username = request.form['username']
#         password_candidate = request.form['password']
#
#         #GET user by Username
#             # PLEASE NOTE: finding users and passwords in the way below is NOT safe and usual
#             # usualy the username calls from a SQL database and password stores as a hash value
#             # therefore it will not be hacked by others.
#             # I used the way below for two reasons:
#             #       1) less complexity at runtime
#             #       2) easy to understand by others
#         if username.lower().strip()=='admin' and password_candidate.lower().strip()=='admin' :
#             #Admin user
#             session['Logged_in'] = True
#             session['username'] = 'admin'
#             session['max_pending_prss'] = 10000
#             return redirect(url_for('home'))
#         elif username.lower().strip()=='armin' and password_candidate.lower().strip()=='12345' :
#             #Armin user
#             session['Logged_in'] = True
#             session['username'] = 'armin'
#             session['max_pending_prss'] = 5
#             return redirect(url_for('home'))
#         elif username.lower().strip()=='sina' and password_candidate.lower().strip()=='abcd' :
#             #Sina user
#             session['Logged_in'] = True
#             session['username'] = 'sina'
#             session['max_pending_prss'] = 3
#             return redirect(url_for('home'))
#         else:
#             flash('Username or Password is wrong. Please try again. (or use "Help me Login!" button!)', 'danger')
#
#
#     return render_template('login.html')
# #end Login page==========================



# #Logout page==========================
# #Logout address that every user can logout with
# @app.route('/logout', methods=['GET', 'POST'])
# @is_logged_in
# def logoutP():
#     session.clear()
#     return redirect(url_for('home'))
# #end Logout page==========================

#upload_db page================================
#Using upload_db, a new CSV dataset uploads in serever and its information stores in "status_manager.csv" file.
@app.route('/upload_db', methods=['GET', 'POST'])
def upload_db():
    #Using upload_dataset a new job adds to status_manager
    if request.method == 'POST':
        # checks if the post request has the file part
        if 'file' not in request.files:
            flash('No file part','danger') # show a warning MSG
            return redirect(url_for('manage_db')) # Go to Homepage
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file','danger') # show a warning MSG
            return redirect(url_for('manage_db')) # Go to Homepage
        if file and allowed_file(file.filename):
            #If every thing is OK with file, uploading process begins
            filename = 'DB' + str(uuid.uuid4().hex[:12]) # Create unique random ID for file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #Save the file in UPLOAD_FOLDER
            file_type = request.form['data_type'] #to specify if it is a ABUNDANCE, UPEX or METADATA file
            file_description = request.form['data_description']
            append_file_id(filename,type=file_type,description=file_description) #to add File_id into status_manager.csv
            flash('Upload is done!','success')
            return redirect(url_for('manage_db'))
        else:
            flash('Only %s file type(s) is allowed.'%(ALLOWED_EXTENSIONS),'danger')
    return render_template('upload_db.html') #for GET requests it just shows an upload page
#end upload_db page============================

#manage_db page================================
@app.route("/manage_db")
def manage_db():
    # Load status_manager to show on the table
    status_manager = pd.read_csv(status_manager_ADDRESS ,header=0,  low_memory=False)#Load all process
    status_manager_records = status_manager.to_records()#Convert dataframe to records for easier use with render_template
    # print(status_manager_records)
    return render_template('manage_db.html',process_list=status_manager_records)
#end Home page=================================

#delete_db page================================
#Using delete_db, a database can be removed from status_manager.csv.
@app.route('/delete_db/<string:file_id>/')
# @is_logged_in
def delete_db(file_id):
    status_manager_df = pd.read_csv(status_manager_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
    status_manager_df = status_manager_df.drop(status_manager_df.index[(status_manager_df["file_id"] == file_id)])# delete the unwanted job
    status_manager_df.to_csv(status_manager_ADDRESS, index=False, header=status_manager_df.columns) #save the new status_manager.csv
    try:
        os.remove(os.path.join('Upload_folder',file_id))
    except:
        print('File did not remove, something went wrong')
    return redirect(url_for('manage_db'))
#end delete_db page=============================

#manage_job page================================
@app.route("/manage_job")
def manage_job():
    # Load process_list to show on the table
    process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
    process_list_records = process_list.to_records()#Convert dataframe to records for easier use with render_template
    abundance_ids, upex_ids, metadata_ids =  find_db_id()
    return render_template('manage_job.html',process_list=process_list_records,abundance_ids=abundance_ids, upex_ids=upex_ids, metadata_ids=metadata_ids)
#end manage_job page============================

#manage_db page================================
@app.route("/download_db/<file_id>/")
def download_db(file_id):
    file_name_download = str(file_id)+'.csv'
    path = os.path.join(app.config['UPLOAD_FOLDER'],file_id)
    return send_file(path, as_attachment=True)
#end Home page=================================

#manage_job page================================
@app.route("/filter_job", methods=['GET', 'POST'])
def filter_job():
    if request.method == 'POST':
        job_type = request.form['job_type']
        abundance_id = request.form['selected_abundance_id']
        upex_id = request.form['selected_upex_id']
        metadata_id = request.form['selected_metadata_id']
        filter_dict = filter_job_listprep(metadata_id)
        return render_template('filter_job.html',filter_dict=filter_dict,abundance_id=abundance_id,upex_id=upex_id,metadata_id=metadata_id,job_type=job_type  )
    return redirect(url_for('manage_job'))
#end manage_job page============================

#submit_job page================================
#Using submit_job, a new processing job is added to Process_list.csv file.
@app.route('/submit_job', methods=['GET', 'POST'])
# @is_logged_in
def submit_job():
    #Using submit_job a new job adds to Process_list
    if request.method == 'POST': #This page is only effective when you POST input_x to it
        data = dict(request.form)
        append_process_id(data)
        # else:
        #     flash('You have reached your limitaion. Please wait till your previous process are done or delete them.', 'warning')
    multiprocess_starter()
    return redirect(url_for('manage_job'))
#end submit_job page============================

#delete_job page================================
#Using delete_job, a Pending job can be removed from Process_list.csv file.
# please note, if a job is processing (by limited_f) or it is Finished, it can NOT delete from the Process_list
@app.route('/delete_job/<job_id>/')
# @is_logged_in
def delete_job(job_id):

    process_job_df = pd.read_csv(process_list_ADDRESS ,header=0, low_memory=False, sep=',')#Load all process
    job_id_index = process_job_df[(process_job_df["job_id"] == job_id)].index.to_list()[0]
    if process_job_df['progress'].loc[job_id_index] == 'Pending...':
        process_job_df = process_job_df.drop(job_id_index)# delete the unwanted job
        process_job_df.to_csv(process_list_ADDRESS, index=False, header=process_job_df.columns) #save the new Process_list.csv
    return redirect(url_for('manage_job'))
#end delete_job page============================

#Home page======================================
@app.route("/")
# @is_logged_in
def home():
    # Load process_list to show on the table
    # process_list = pd.read_csv(process_list_path ,header=0,  low_memory=False, sep=',')#Load all process
    #count_pending_jobs in order to show up on the page
    # count_user_pending_process = count_pending_jobs(session['username'])
    #Convert dataframe to records for easier use with render_template
    # process_list_records = process_list.to_records()
    return render_template('home.html')#,count_user_pending_process=count_user_pending_process,process_list=process_list_records)
#end Home page==================================

#sigmeta_list page==============================
# Presenting Signaling Metabolite Results
@app.route("/sigmeta_list")
def sigmeta_list():
    # Load process_list to show on the table
    process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
    process_list = process_list[['job_id','submit_date','model_type','progress','comparison_factor','description']]
    process_list = process_list[process_list['progress']=='Finished']
    process_list = process_list[process_list['model_type']=='signaling_meta']
    process_list_records = process_list.to_records()#Convert dataframe to records for easier use with render_template
    return render_template('sigmeta_list.html',process_list=process_list_records)
#end sigmeta_list page==========================

#bacteria_list page==============================
# Presenting Bacteria Influences Results
@app.route("/bacteria_list")
def bacteria_list():
    # Load process_list to show on the table
    process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
    process_list = process_list[['job_id','submit_date','model_type','progress','comparison_factor','description']]
    process_list = process_list[process_list['progress']=='Finished']
    process_list = process_list[process_list['model_type']=='bac_influence']
    process_list_records = process_list.to_records()#Convert dataframe to records for easier use with render_template
    return render_template('bacteria_list.html',process_list=process_list_records)
#end sigmeta_list page==========================

#sigmeta_excel page==============================
# Download Signaling Metabolite Excel file
@app.route("/sigmeta_excel/<path:job_id>/")
def sigmeta_excel(job_id):
    # Load process_list to show on the table
    try:
        file_name = job_id.strip()+str('.xlsx')
        file_address = os.path.join(app.config['OUTPUT_FOLDER'],file_name)
        print(file_address)
        return send_file(file_address, as_attachment=True)
    except:
        flash('File not found','danger')
        return redirect(url_for('sigmeta_list'))
#end sigmeta_excel page==========================

#paretoplot page=================================
# Download Signaling Metabolite Excel file
@app.route("/paretoplot/<job_id>/")
def paretoplot(job_id):
    # Load process_list to show on the table
    return render_template('paretoplot.html',job_id=job_id)
@app.route('/paretocsv/<job_id>/')
def csv_d3(job_id):
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'],'Paretoplot','%s.csv'%(job_id)))
#end paretoplot page=============================

#networkshow page==============================
# Download Signaling Metabolite Excel file
@app.route("/networkshow/<net_type>/<job_id>/")
def networkshow(net_type,job_id):
    return render_template('networkshow.html',net_type=net_type,job_id=job_id)
@app.route('/networkjson/<net_type>/<job_id>/')
def networkjson(net_type,job_id):
    if net_type=='sp_meta':
        path_json = os.path.join(app.config['OUTPUT_FOLDER'],'Network_Normal','%s.json'%(job_id))
    elif net_type=='con_meta':
        path_json = os.path.join(app.config['OUTPUT_FOLDER'],'Network_Effective','%s.json'%(job_id))
    return send_file(path_json)
#end networkshow page==========================

#Test page=======================================
@app.route("/test")
# @is_logged_in
def testpage():

    job_id = 'JOB773553773'

    # process_list = pd.read_csv(process_list_ADDRESS ,header=0, index_col=0, low_memory=False, sep=',')#Load all process
    # abundance_id = process_list['abundance'].loc[job_id]
    # upex_id = process_list['upex'].loc[job_id]
    # metadata_id = process_list['metadata'].loc[job_id]
                                                                        #abundance_id
    abundance_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'DBffb96ab62ae0') ,header=0, index_col=0)
    upex_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'DBcb4d26fe0faf') ,header=0, index_col=0)
    metadata_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'DBa2017f963c12') ,header=0, index_col=0)

    filter_params = {

    }
    comp_fact = ''
    comp_fact_value = comparison_combine_v3_OneDim(comp_fact,'DBa2017f963c12',filter_params)

    #TODO first filter abundance data
    Bacteria_Influence_v1(job_id, abundance_df,upex_df,metadata_df,filter_params,comp_fact, comp_fact_value)

    # file_exceler_SigMeta(job_id)

    return redirect(url_for('home'))
#end Test page==================================

#end Web Pages **********************************************************





# Functions *************************************************************
# append_file_id====================================
def append_file_id(fileID,type,description):
    '''
    Append information of newly uploaded file in "status_manager.csv"

    Input(s):
    fileID: The unique random ID which is created after the file is uploaded

    Output(s):
    -
    '''

    data_new_row = {
    'file_id': fileID,
    'upload_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'type': type,
    'description': description,
    }
    df_new_row = pd.DataFrame(data_new_row, index=[0])
    df_new_row.to_csv(status_manager_ADDRESS, mode='a', index=False, header=False) #saves status_manager.csv
#end append_file_id=================================

# append_process_id=================================
def append_process_id(data):
    '''
    Append information of newly uploaded file in "process_list.csv"

    Input(s):
    data: The Dictionary received from filter_job page

    Output(s):
    -
    '''
    filter_params = data.copy()
    not_filter_keys = ['selected_abundance_id','selected_upex_id',
            'selected_metadata_id','comparison_factor',
            'description']
    for key in not_filter_keys:
        filter_params.pop(key, None)
    # count_user_pending_process  = count_pending_jobs(session['username'])# number of pendig process for the logged in user
    # if count_user_pending_process < session['max_pending_prss']: # Checks if the user is able to submit a new job or not
    process_job = { #New job
        'job_id' : 'JOB' + str(uuid.uuid4())[:13],
        'submit_date' : dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type' : data['job_type'],# this is either 'signaling_meta' or 'bac_influence'
        'progress' : 'Pending...',
        'abundance' : data['selected_abundance_id'],
        'upex' : data['selected_upex_id'],
        'metadata' : data['selected_metadata_id'],
        'comparison_factor': data['comparison_factor'],
        'filter_params' : str(filter_params),
        'description' : data['description']
     }

    process_job_df = pd.DataFrame(process_job , index=[0])#Load all process
    process_job_df.to_csv(process_list_ADDRESS, mode='a', index=False, header=None) #store the new job into the Process_list
#end append_file_id=================================

# allowed_file======================================
def allowed_file(filename):
    '''
    Checks if the uploaded file has aد appropriate extension

    Input(s):
    filename: The file name when it is uploading

    Output(s):
     -  -  - :(Boolean) Basically it says Yes or NO
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#end allowed_file===================================

# Find Database's ID================================
def find_db_id():
    '''
    finds ID for each ABUNDANCE, UPEX and METADATA database, so that
    it can be used for "Submit New Job" form.

    Input(s):

    Output(s):
     abundance_ids  : ID and Description for all ABUNDANCE Database
     upex_ids     : ID and Description for all UPEX Database
     metadata_ids : ID and Description for all UPEX Database
    '''
    status_manager_df = pd.read_csv(status_manager_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all databases

    status_manager_df_abundance = status_manager_df[status_manager_df['type']=='abundance']
    abundance_ids = [str(status_manager_df_abundance['file_id'].iloc[x])+' - '+
                    str(status_manager_df_abundance['description'].iloc[x]) for x in range(len(status_manager_df_abundance)) ]

    status_manager_df_upex = status_manager_df[status_manager_df['type']=='upex']
    upex_ids = [str(status_manager_df_upex['file_id'].iloc[x])+' - '+
                    str(status_manager_df_upex['description'].iloc[x]) for x in range(len(status_manager_df_upex)) ]

    status_manager_df_metadata = status_manager_df[status_manager_df['type']=='metadata']
    metadata_ids = [str(status_manager_df_metadata['file_id'].iloc[x])+' - '+
                    str(status_manager_df_metadata['description'].iloc[x]) for x in range(len(status_manager_df_metadata)) ]

    return abundance_ids, upex_ids, metadata_ids
#end allowed_file===================================

# filter_job_listprep===============================
def filter_job_listprep(metadata_id):
    '''
    Prepares a list to create filter_job page. finds different types of
    filter parameters and accpetable values based on chosen META-Data.

    Input(s):
    metadata_id: The file name of META-Data

    Output(s):
     filter_dict : A dictionary to show information on the page
    '''

    metadata_ADDRESS = os.path.join(app.config['UPLOAD_FOLDER'],metadata_id.strip())
    metadata_df = pd.read_csv(metadata_ADDRESS ,header=0, index_col=0)#Load META-Data

    dict_keys = metadata_df.columns.to_list()

    filter_dict = {
        dict_keys[x] : unique(metadata_df[dict_keys[x]].to_list()) for x in range(len(dict_keys))
    }

    return filter_dict
#end filter_job_listprep============================

# comparison_combine================================
def comparison_combine_v3_OneDim(comparison_factor,metadata_id ,filter_params):
    '''
    Combine different situations for calculating Signaling Metabolite

    Input(s):
    comparison_factor: a parameter that Signaling Metabolite should be compared by

    Output(s):
     combined_comparison_factor : combined comparison_factor two by two
    '''
    if (comparison_factor != 'ONE_DIMENSIONAL') or (comparison_factor != ''):
        metadata_ADDRESS = os.path.join(app.config['UPLOAD_FOLDER'],metadata_id.strip())
        metadata_df = pd.read_csv(metadata_ADDRESS ,header=0, index_col=0)#Load META-Data

        metadata_df_filtered = filter_data_aundance_v6_OneDim(pd.DataFrame(), metadata_df, filter_params, None, None)

        columns = metadata_df_filtered.columns.to_list()

        if comparison_factor in columns:
            combined_comparison_factor = combine(unique(metadata_df_filtered[comparison_factor].to_list()))
        else:
            combined_comparison_factor = None
    else: # comparison_factor == 'ONE_DIMENSIONAL'
        combined_comparison_factor = ['ONE_DIMENSIONAL','ONE_DIMENSIONAL']#two 'ONE_DIMENSIONAL' because in datadiff process we need two of them to
                                                                          #find datadiff1 and datadiff2

    return combined_comparison_factor
#end comparison_combine=============================

# combine===========================================
def combine(list1):
    '''
    Combine  list element two by two. it is useful for combining
    comparison_factor if they are more than two elements.

    Input(s):
    list1: a list to combine elements two by two ex. ['a','b','c']

    Output(s):
     combined_list : combined list ex. [['a','b'],['a','c'],['b','c']]
    '''

    if len(list1)<2:
        combined_list = None #input values is not acceptable
    else:
        combined_list = []
        for i in range(len(list1)):
            for j in range(i, len(list1)):
                if i==j:
                    continue
                combined_list.append([list1[i],list1[j]])

    return combined_list
#end combine========================================

# limited_f=========================================
def limited_f(job_id): #this function just runs 3 chained for-loops and then returns the input_x
    # try:
    process_list = pd.read_csv(process_list_ADDRESS ,header=0, index_col=0, low_memory=False, sep=',')#Load all process
    abundance_id = process_list['abundance'].loc[job_id].strip()
    upex_id = process_list['upex'].loc[job_id].strip()
    metadata_id = process_list['metadata'].loc[job_id].strip()
    filter_params = eval(process_list['filter_params'].loc[job_id])
    comp_fact = process_list['comparison_factor'].loc[job_id]#.strip()
    if comp_fact==None:
        comp_fact = ''
    else:
        comp_fact = str(comp_fact).strip()
    job_type = process_list['model_type'].loc[job_id].strip()

    abundance_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],abundance_id) ,header=0, index_col=0)
    upex_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],upex_id) ,header=0, index_col=0)
    metadata_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],metadata_id) ,header=0, index_col=0)


    comp_fact_value = comparison_combine_v3_OneDim(comp_fact, metadata_id,filter_params)

    if job_type=='signaling_meta':
        Signaling_Metabolite_v9(job_id, abundance_df,upex_df,metadata_df,filter_params,comp_fact, comp_fact_value)
        file_exceler_SigMeta(job_id)
    elif job_type=='bac_influence':
        print('I am at bac_influence Algorithm')
        Bacteria_Influence_v1(job_id, abundance_df,upex_df,metadata_df,filter_params,comp_fact, comp_fact_value)


    # except:
    #     failed_jobs_correction()

    return job_id
#end limited_f======================================

# count_pending_jobs================================
def count_pending_jobs(user='ALL'):
        ''' This function counts every pending jobs for each users

        --INPUT:
            - user : the username that we need to know the number of pending jobs

        --OUTPUT:
            - count_user_pending_process : the number of  pending jobs

        '''
        #Load every process (either Finished or Pending)
        process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
        #Number of Pending process for the logged in user
        if user == 'ALL':
            #if user is 'ALL' it will count all pending... and working jobs together
            count_user_pending_process = len(process_list[(process_list['progress']=='Pending...')]) + len(process_list[(process_list['progress']=='Working')])
        else:
            process_list = process_list[(process_list['progress']=='Pending...')]
            count_user_pending_process = len(process_list[process_list['by']==session['username']])

        return int(count_user_pending_process)
#end count_pending_jobs=============================

#multiprocess_starter===============================
def multiprocess_starter():
    #This function uses Multiprocessing library to start running limited_f Function
    # In fact, this function call multiprocess_keep_running function as a multiprecessing object.
    # On the other hand, multiprocess_keep_running calls limited_f and finishes every pending process.
    # The main goal for this function is to prevent creating more than one multiprecessing object and
    # delete it when the every pending process is done.
    global MULTIPROCESS_list
    all_pending_and_working_jobs = count_pending_jobs('ALL') #counts every Pending... and Working jobs
    print('Hey Im at starter',MULTIPROCESS_list,all_pending_and_working_jobs)
    if (all_pending_and_working_jobs > 0):
        if (not MULTIPROCESS_list): #This means there is no active Multiprocessing objects
            print('Hey I start it right now')
            myProcess = mp.Process(target=multiprecess_keep_runing) #Create a multiprecess object
            MULTIPROCESS_list= myProcess#stores if a process is runing already
            myProcess.start()#start the multiprecessing
        else:
            if not MULTIPROCESS_list.is_alive():
                print('Hey I start it right now')
                myProcess = mp.Process(target=multiprecess_keep_runing) #Create a multiprecess object
                MULTIPROCESS_list= myProcess#stores if a process is runing already
                myProcess.start()#start the multiprecessing
#end multiprocess_starter===========================

#multiprecess_keep_runing===========================
def multiprecess_keep_runing():
    #This function runs in Background (as a multiprecessing object). it contains a while-loop
    # to find every pending process and run limited_f for each one of them.
    # when every pending process finished successfully, it calls multiprocess_starter Function
    # to remove multiprocessing object from memory and free hardware resources.

    # First finds number of Pending processes
    process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
    pending_list = process_list[(process_list['progress']=='Pending...')]
    all_pending_jobs = len(pending_list)

    while all_pending_jobs > 0:
        print('Remaining Pendings...',all_pending_jobs)
        first_pending_job_id = pending_list['job_id'].iloc[0] #Find process_id for the oldest Pending process
        process_list.at[process_list.index[(process_list["job_id"] == first_pending_job_id)], 'progress'] = 'Working' #Change its progress to Working
        process_list.to_csv(process_list_ADDRESS, index=False, header=process_list.columns) #save the changes


        input_liminted_f = first_pending_job_id#find the input_x for the oldest Pending process
        output_f = limited_f(input_liminted_f) # Runs limited_f


        process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
        process_list.at[process_list.index[(process_list["job_id"] == first_pending_job_id)], 'progress'] = 'Finished' #Change its progress to Finished
        process_list.to_csv(process_list_ADDRESS, index=False, header=process_list.columns) #save the changes
        time.sleep(3)
        pending_list = process_list[(process_list['progress']=='Pending...')] #finds the number of remaining Pending process

        # pending_list = process_list[(process_list['progress']=='Pending...')] #finds the number of remaining Pending process
        all_pending_jobs = len(pending_list)
        print('All Pending Jobs: ',all_pending_jobs)
    #After all Pending processes are done, multiprocess_starter calls to free up memory and CPU
    multiprocess_starter()
#end multiprecess_keep_runing=======================

# failed_jobs_correction====================================
def failed_jobs_correction():
    '''
    This function should run right at the begining to chnage every "Working"
    process to "Failed"

    Input(s):
    -

    Output(s):
    -
    '''

    process_list = pd.read_csv(process_list_ADDRESS ,header=0,  low_memory=False, sep=',')#Load all process
    working_list = process_list[(process_list['progress']=='Working')]


    if len(working_list)>0:
        job_id_list = working_list['job_id'].to_list()
        # print(job_id_list)
        for each_job_id in job_id_list:
            process_list.at[process_list.index[(process_list["job_id"] == each_job_id)], 'progress'] = 'Failed'
        process_list.to_csv(process_list_ADDRESS, index=False, header=process_list.columns) #save the changes
#end failed_jobs_correction=================================
#end Functions **********************************************************





if __name__ == '__main__':
    failed_jobs_correction()
    app.run(debug=True,host="0.0.0.0", port=5000)
