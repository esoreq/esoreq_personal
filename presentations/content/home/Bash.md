
# Some bash Basics

- In this section we will cover how to create custom commands (aliases), allowing you to create shortcuts for a single command or a group of commands.
- We will also learn how to configure your personal home account on the cluster 
- How to transfer files from and to the cluster 
- Finally how to extend the cluster functionality using modules 

//slide//

# Intro

- The bash Shell (Bourne Again SHell), is a command-line interface (CLI). Lets go over the most popular commands, and learn how to use them. 

- If you want to extend your understanding you should consider taking [Learn bash the Hard Way](https://www.educative.io/courses/master-the-sh-shell).

//slide//

# Intro

- However, for our needs the basics we will cover this week should be enough

- If you just want to know more on a specific issue here are some websites that can be useful: 
    - [sh-shell-command-cheat-sheet](https://www.educative.io/blog/sh-shell-command-cheat-sheet)
    - [TecMint](https://www.tecmint.com/)
    - [Advanced sh-Scripting Guide](https://www.tldp.org/LDP/abs)

- Finally here is a good [Cheat sheet](https://devhints.io/sh) to explore when trying to understand my functions 



//slide//
## bash Basics
 
- Lets go over some important commands and how to use them.
- Almost all commands in bash will have some help option (try it!)

```bash
echo --help # Open the help page of the command 
```


//slide//

##  cd,pwd - Change directory and Print working directory

### Examples 

```bash
cd Mail # change the directory 
pwd # print the current directory. 
cd /  # change directory to the root directory
cd ~  # change directory to your home directory
cd dir_root/dir_branch/dir_leaf # goto leaf dir
cd .. # go back one level 
cd "dir name" 
# if for some reason you created a folder name with spaces 
# (DONT!!!) this is how you call it 
```


//slide//

##  mkdir - Create a directory

The mkdir command in UNIX allows users to create directories aka folders. 

### Examples 

```bash
mkdir ~/sandbox   
# create a folder named sandbox in your home directory
cd ~/sandbox
mkdir dir_1 dir_2  
# create multiple directories 
mkdir -p dir_root/dir_branch/dir_leaf 
# create Folder structure 
ls -R dir_root/ # confirm using ls
mkdir -p dir_root/{dir_b}/{dir_l1,dir_l2} 
# This supports complex assignments
mkdir -vp dir_root/{dir_b1,dir_b2}/{dir_l1,dir_l2} 
# v verbosely report terminal assignments 
```


//slide//

## echo  

- Prints text to the terminal window
- The most basic process in any programming language is the ability to out put information.
- The echo command prints text to the terminal window and is typically used in shell scripts and batch files to output status text to the screen or a computer file. 
- Echo is also particularly useful for showing the values of environmental variables, which tell the shell how to behave as a user works at the command line or in scripts.

//slide//

## echo 

### Examples 

```bash
cd ~/sandbox
echo Some text to print out 
# any string after the command will be printed 
echo * 
#  Will Print all the files/folder using echo command 
echo *.jpeg 
#  Will Print all the files of a specific kind
echo "Some text to add to a file" > file  
# Use redirect operator > to store output to file
cat file
# the cat command will read and output the files content 
```

//slide//

## cat — Read a file, create a file, and concatenate files

Cat (short for “concatenate“) is one of the more versatile commands and serves three main functions:

- Displaying files
- Combining multiple files
- Creating new files.


//slide//

## cat 

### Examples 

```bash
cat file 
# Displaying Contents of a File
cat > file_2 
# Allows to write text interactively 
cat file_2 > file 
# It is possible to redirect > to output like echo 
# However the above command overwrites the file contents 
cat file file_1 file_2 
# We can also view contents of multiple files
cat file_1 >> file 
# file_1 content will be appended at the end of file
```


//slide//

## Variables in sh

- A variable is a temporary store for a piece of information. 
- Essentially, all bash variables are character strings, however, depending on context, it is possible to run arithmetic operations and comparisons on variables.
- As we are not going to write fancy bash scripts I will not go over arithmetics, however, check this [link](https://sh.cyberciti.biz/guide/Perform_arithmetic_operations#Arithmetic_Expansion_in_sh_Shell) for examples.  


//slide//

## Variables -  Examples 

```bash
a = 10 # spaces are not allowed in variable assignment
a=10 # This will assign the string 10 to a 
b=a # Assign the string a to b 
echo $a # use the $ sign to call
b=$a # assign a to b 
let a+=10 # Add ten to a 
echo $a $b # confirm
```



//slide//

## ls — (List directory contents)

- Probably the most common command. The **ls** command allows you to quickly view all files within the specified directory.
- Like almost all bash commands it can be extended by adding different options
- Let's review some useful ones

//slide//

## ls — Examples 

```bash
ls       # list files and directories in bare format 
ls -a    # List all files including hidden file
ls -F    # Will distinguish folders from files
ls -l    # Shows additional content 
ls -lh   # human readable format.
ls -R    # Recursively list Sub-Directories 
ls -1    # Force output to one column 
```



//slide//

##  cp,mv - copy, move or rename files and directories

### Examples 

```bash
cd ~ # make sure we are home 
output=dir_root/dir_branch/dir_leaf 
# make a variable to store path
cp file $output # copy file to path
mv file_1 $output # move file_1 to output
ls $output # list files in output
mv file $output/file_renamed 
# move file to output and rename it 
mv -t dir dir_1 dir_2 
# move folders to --target-directory (long form of -t)
cp -R dir $output 
# recursively copy folder content to path
ls -FR $output # list files in output again
```

//slide//

## history

The GNU history command keeps a list of all the other commands that have been run from that terminal session, then allows you to replay or reuse those commands instead of retyping them. 


//slide//

## history, pipe

A pipe is a form of redirection (transfer of standard output to some other destination) that is used in Linux and other Unix-like operating systems to send the output of one command/program/process to another command/program/process for further processing.
You can make it do so by using the pipe character '|'.

//slide//

## history, pipe and grep 

Global regular expression print (grep for short) is a command-line utility for searching plain-text data sets for lines that match a regular expression. Its name comes from the ed command g/re/p, which has the same effect: doing a global search with the regular expression and printing all matching lines.

//slide//

## history, pipe and grep - Examples 

```bash
history # will print out all the commands you made 
echo $HISTTIMEFORMAT  
# HISTTIMEFORMAT controls how history is printed 
# %Y-%m-%d %H:%M:%S = Year month day hour minute second 
export HISTTIMEFORMAT='' 
# we can remove the formating to have only the commands 
history | more 
# The | symbol stands for piped commands 
history | grep ls 
# Will return any command with ls 
history -c 
# Will clear the history file 
```

//slide//
 
## rm - remove files and folders

- As the name suggests rm command is used to delete or remove files and directories
- If you are new to bash then you should be very careful when using the rm command
- In linux once you delete the files you can not recover the contents of files and directory

//slide//
 
## rm - Examples 

```bash
rm file_2 # delete file 
mkdir dir_empty
rm dir_empty 
# will fail 
rm -d dir_empty 
# delete folder if it's empty
rm -r dir 
# recursively delete folder and it's contents 
rm -ir dir_root 
# recursively delete contents and prompt before every removal
```

//slide//

## du - disc usage

- The du command, is used to command estimates file_path space usage.

### Examples 

```bash
du -sh ~ 
# Human readable format home directory space usage.
du -sh -- * 
# cumulative disk usage of all non-hidden directories
du -hs $(ls -A) 
# Will include dotfiles (normally hidden from you)
```


//slide//

## Summary 

- These are the most common commands you are likely to use 
- However, during the course we will learn additional ones
- This is a good opportunity to go over the various commands and see that you understood everything 
- Also on a personal note, please make note for anything that isn't clear so I can learn from your experience
- Time to make a cup of coffee and take a short break 
- When done go to the advanced sh section





