import os    # for file handling  
import sys   # for writing/printing messages
import math

import re    # for splitting input

from file_utils import remove_file, rename_file, DISK_PATH

######### CONSTANTS (unlimited number allowed) #######################
PATH = os.path.join(DISK_PATH, "") # path to store files (DO NOT CHANGE)
SEP = "====" # used to separate messages when printing them out

######### VARIABLES (at most 10 active at any time) ##################
# the current file that is available to add to  
cur_file = 1
# the number of messages that are currently in the system
msg_count = 0
# how many messages are there in the working file right now
msg_in_file = 1
# name of person who is posting
poster = ""
# maximum number of messages allowed (usually 200)
max_msg_count = 200
# maximum number of messages allowed in a file (usually 10)
max_msg_per_file = 10
# a tracker for ids that are made available from remove_msg
avail_ids = ""

######### EXCEPTIONS #################################################
class MessagesFullExn(Exception):
    pass

######### SYSTEM SETUP, SHUTDOWN, AND RESET ##########################
def connect(username: str, restart: bool) -> None:
    """
    Starts a connection to the system by the named user

    Parameters:
    username -- the name of the user who is connecting (they will be the
                poster of messages added until they disconnect)
    restart -- if the program has just connected to the server
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids

    # if a disk folder does not exist, make one
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    
    # update username regardless of soft disconnect/disconnect
    poster = username

    # update all the variables if a disconnect has happened
    backup = os.path.join(PATH, "backup.txt")
    if restart and os.path.exists(backup):
        with open(backup, "r") as input:
            cur_file = int(input.readline().strip())
            msg_count = int(input.readline().strip())
            msg_in_file = int(input.readline().strip())
            max_msg_count = int(input.readline().strip())
            max_msg_per_file = int(input.readline().strip())
            avail_ids = input.readline().strip()
        remove_file(backup)
                

def disconnect() -> None:
    """
    Disconnects the current user (this will depend on your design) and saves
    data as necessary so that the system can resume even if the Python program
    is restarted 
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids

    # save global information locally so info can be backed up
    cur_file_l = cur_file
    msg_count_l = msg_count
    msg_in_file_l = msg_in_file
    max_msg_count_l = max_msg_count
    max_msg_per_file_l = max_msg_per_file
    avail_ids_l = avail_ids

    # clear all globals
    reset_stats(200, 10)

    # write the variables to a new file
    backup_path = os.path.join(PATH, "backup.txt")
    with open(backup_path, "w") as f:
        f.write(str(cur_file_l) + "\n")
        f.write(str(msg_count_l) + "\n")
        f.write(str(msg_in_file_l) + "\n")
        f.write(str(max_msg_count_l) + "\n")
        f.write(str(max_msg_per_file_l) + "\n")
        f.write(avail_ids_l)
        

def soft_disconnect() -> None:
    """
    Disconnects the current user (this will depend on your design)
    """
    global poster

    poster = ""
    

def clean_reset(msg_max_val=200, msg_per_file_val=10) -> None:
    """
    Deletes all the disk files to start a clean run of the system.
    Supports setting different constant values.
    Useful for testing.

    Parameters:
    msg_max_val -- max number of messages system can hold
    msg_per_file_val -- max number of messages each file can hold

    """
    
    # TODO: Fill in with what makes sense for your design.
    # It might relate to how you store your necessary info
    # between (dis)connections to the server!
    # Feel free to pass in different values when testing for clean_reset,
    # 200 and 10 are just the default. (You do not need to edit
    # the method header to do this. Just pass different values in when calling)

    # reset all the stats of the system
    reset_stats(msg_max_val, msg_per_file_val)

    # clean the disk
    if os.path.exists(PATH):
        all_files = os.scandir(PATH)
        for file in all_files:
            remove_file(file)
    

######## DESIGN HELPERS ##########################################
def write_msg(f, id: int, who: str, subj: str, msg: str, labeled=False) -> str:
    """
    Writes a message to the given file handle. e.g., If you want to print to a
    file, open the file and use fh from the following code as the first argument

           with open(FILENAME, mode) as fh

    If you want to print to the console/screen, you can pass the following as 
    the first argument

            sys.stdout

    msg can be passed as false to suppress printing the text/body of the message.

    Parameters:
    f -- file descriptor
    id -- message id
    who -- poster
    subj -- subject line
    msg -- body text
    labeled -- boolean deciding if labels should also be used
    """
    result = ""
    
    if labeled:
        f.write("ID: " + str(id) + "\n")
        result += "ID: " + str(id) + "\n"

        f.write(who + "\n")
        result += who + "\n"

        f.write(subj + "\n")
        result += subj + "\n"

        if msg: 
            f.write(msg + "\n")
            result += msg + "\n"
    else: # needs labels
        f.write(SEP + "\n")
        result += SEP + "\n"

        f.write("ID: " + str(id) + "\n")
        result += "ID: " + str(id) + "\n"
        
        f.write("Poster: " + who + "\n")
        result += "Poster: " + who + "\n"

        f.write("Subject: " + subj + "\n")
        result += "Subject: " + subj + "\n"

        if msg: 
            f.write("Text: " + msg + "\n")
            result += "Text: " + msg + "\n"
    
    return result


def split_string_exclude_quotes(s) -> list[str]:
    """
    Splits a given string and splits it based on spaces, while also grouping
    words in double quotes together.

    Parameters:
    s -- string to be split
    Returns:
    A list of strings after splitting
    Example:
    'separate "these are together" separate` --> 
    ["separate", "these are together", "separate"]
    """
    # This pattern matches a word outside quotes or captures a sequence of 
    # characters inside double quotes without including the quotes
    pattern = r'"([^"]*)"|(\S+)'
    matches = re.findall(pattern, s)
    # Each match is a tuple, so we join non-empty elements
    return [m[0] if m[0] else m[1] for m in matches]

def get_file_index(id: int) -> int:
    """
    get a file index from a given ID

    Parameters:
    id -- message ID
    Returns:
    The integer file index that corresponds to the id
    """
    if (id % 10 == 0):
        return int(id / 10 - 1)
    else:
        return math.floor(id / 10)

def print_stats():
    """
    Prints all the stats in the system for testing
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids

    print("cur file: " + str(cur_file))
    print("msg_count: " + str(msg_count))
    print("msg_in_file: " + str(msg_in_file))
    print("poster: " + poster)
    print("max_msg_count: " + str(max_msg_count))
    print("max_msg_per_file: " + str(max_msg_per_file))
    print("avail_ids: " + avail_ids)

def reset_stats(msg_max_val=200, msg_per_file_val=10):
    """
    Helper to clean_reset(), only resets global variables to default

    Parameters:
    msg_max_val -- max number of messages system can hold
    msg_per_file_val -- max number of messages each file can hold
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids

    # reset all globals
    msg_count = 0
    msg_in_file = 0
    max_msg_count = msg_max_val
    max_msg_per_file = msg_per_file_val
    cur_file = 1
    poster = ""
    avail_ids = ""

def remove_who_subj(id: int) -> None:
    """
    Removes a record of the message's subj/who in summaries.txt so
    print_summary() cannot access it

    Parameters: message id
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids

    # path of the file from which the msg will be removed
    file_path = os.path.join(PATH, "summary.txt")
    # temporary file created for msg removal
    temp_path = os.path.join(PATH, "temp_sum.txt")

    # write everything from origial file with msg to the temp file except for
    # the msg itself
    with open(temp_path, "w") as output:
        with open (file_path, "r") as input:
            for line in input: 
                if line == "ID: " + str(id) + "\n":
                    input.readline()
                    input.readline()
                else:
                    output.write(line)
               
    remove_file(file_path)
    rename_file(temp_path, file_path)
    

####### CORE SYSTEM OPERATIONS ####################################
def show_menu(): 
    """
    Prints the menu of options.
    """
    print("Please select an option: ")
    print("  - type A <subj> <msg> to add a message")
    print("  - type D <msg-num> to delete a message")
    print("  - type S for a summary of all messages")
    print("  - type S <text> for a summary of messages with <text> in title or poster")
    print("  - type V <msg-num> to view the contents of a message")
    print("  - type X to exit (and terminate the Python program)")
    print("  - type softX to exit (and keep the Python program running)")


def post_msg(subj: str, msg: str) -> None:
    """
    Stores a new message (however it makes sense for your design). Your code
    should determine what ID to use for the message, and the poster of the
    message should be the user who is connected when this function is called

    Parameters:
    subj -- subject line
    msg -- message body
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids
    
    # check if num of current stored exceeds max possible stored message
    if (msg_count + 1 > max_msg_count or
        len(subj) > 32 or
        len(msg) > 128):
        raise(MessagesFullExn())

    else:
        # update overall msg count
        msg_count += 1

        # adjust file stats based on available ids from removal and write msg
        if avail_ids != "":
            # get the first of the available ids
            avail_id_str = split_string_exclude_quotes(avail_ids)[0]
            # type cast to int
            avail_id = int(avail_id_str)
            # file to remove from
            filename = os.path.join(PATH, str(get_file_index(avail_id)) 
                                    + ".txt")
            
            # update avail_ids by removing the space that will be filled 
            remove_sub = len(avail_id_str) + 1
            avail_ids = avail_ids[remove_sub:]
            
            # post the message
            with open(filename, "a") as f:
                write_msg(f, avail_id, poster, subj, msg, labeled=False)
            
            # write poster and subj to summary.txt
            with open(os.path.join(PATH, "summary.txt"), "a") as f:
                write_msg(f, avail_id, poster, subj, None, labeled=True)
        else: 
            msg_in_file += 1 

            # one file is full, advance current file
            if msg_in_file > max_msg_per_file:
                cur_file += 1
                msg_in_file = 1
            
            filename = os.path.join(PATH, str(get_file_index(msg_count)) 
                                    + ".txt")

            # post the message
            with open(filename, "a") as f:
                write_msg(f, msg_count, poster, subj, msg, labeled=False)

            # write poster and subj to summary.txt
            with open(os.path.join(PATH, "summary.txt"), "a") as f:
                write_msg(f, msg_count, poster, subj, None, labeled=True)

def find_print_msg(id: int) -> str:
    """
    Prints contents of message for given ID. 

    Parameters:
    id -- message ID
    Returns:
    The string to be printed (for autograder).
    """
    result = ""
    file_index = get_file_index(id)
    file_name = os.path.join(PATH, str(file_index)) + ".txt"
    with open(file_name, "r") as input:
        for line in input:
            if line.strip() == "ID: " + str(id):
                result += (SEP + "\n")
                result += line
                result += input.readline()
                result += input.readline()
                result += input.readline().strip()
    
    print(result)
    return result


def remove_msg(id: int) -> None:
    """
    Removes a message from however your design is storing it. A removed message
    should no longer appear in summaries, be available to print, etc.
    """
    global cur_file
    global msg_count 
    global msg_in_file
    global poster
    global max_msg_count 
    global max_msg_per_file 
    global avail_ids

    # path of the file from which the msg will be removed
    file_path = os.path.join(PATH, str(get_file_index(id)) + ".txt")
    # temporary file created for msg removal
    temp_path = os.path.join(PATH, "temp.txt")

    # write everything from origial file with msg to the temp file except for
    # the msg itself
    with open(temp_path, "w") as output:
        with open (file_path, "r") as input:
            for line in input:
                if line == SEP + "\n": 
                    id_line = input.readline() 
                    if id_line == "ID: " + str(id) + "\n":
                        input.readline()
                        input.readline()
                        input.readline()
                    else:
                        output.write(SEP + "\n")
                        output.write(id_line)
                else:
                    output.write(line)
    remove_file(file_path)
    rename_file(temp_path, file_path)

    # fully remove record of message from summary.txt
    remove_who_subj(id)

    # update global stats
    avail_ids += (str(id) + " ")
    msg_count -= 1
                

def print_summary(term = "") -> str:
    """
    Prints summary of messages that have the search term in the who or subj fields.
    A search string of "" will match all messages.
    Summary does not need to present messages in order of IDs.

    Returns:
    A string to be printed (for autograder).
    """
    msg = ""
    filename = os.path.join(PATH, "summary.txt")
    if os.path.exists(filename):
        with open(filename, "r") as input:
            for line in input:
                if ("ID: " in line):
                    who = input.readline().strip()
                    subj = input.readline().strip()
                    if (term in who or term in subj):
                        id = int(split_string_exclude_quotes(line)[1])
                        msg += write_msg(sys.stdout, id, who, subj, None, False)
    return msg


############### SAMPLE FROM HANDOUT ######################

# Our test cases will look like this, with assertions intertwined

def sample():
    connect("kathi", True)
    post_msg("post homework?", "is the handout ready?")
    post_msg("vscode headache", "reinstall to fix the config error")
    soft_disconnect() # keep the python programming running and connect another user
    connect("nick", False)
    print_summary("homework")
    find_print_msg(1)
    post_msg("handout followup", "yep, ready to go")
    remove_msg(1)
    print_summary()
    disconnect()

############### MAIN PROGRAM ############################

# If you want to run the code interactively instead, use the following:

def start_system():
    """
    Loop to run the system. It does not do error checking on the inputs that
    are entered (and you do not need to fix that problem)
    """
    
    print("Welcome to our BBS!")
    print("What is your username?")
    connect(input(), True)

    done = False
    while(not done):
        show_menu()
        whole_input = input() # read the user command
        choice = split_string_exclude_quotes(whole_input) #split into quotes
        match choice[0].upper():
            case "A": 
                try:
                    post_msg(choice[1], choice[2]) # subject, text
                except:
                    print("Error: maximum capacity reached. To post, " +
                          "shrink your subject or message or delete a " + 
                          "previous message.")
                    show_menu()
            case "D": 
                remove_msg(int(choice[1]))
            case "S": 
                if len(choice) == 1:
                    print_summary("")
                else:
                    term = choice[1]
                    print_summary(term)
            case "V":
                find_print_msg(int(choice[1]))
            case "X": 
                disconnect()
                done = True
                exit()
            case "SOFTX":
                soft_disconnect()

                # restart menu 
                print("What's your username?")
                connect(input(), False)
            case _: 
                print("Unknown command")

# uncomment next line if want the system to start when the file is run
# start_system()