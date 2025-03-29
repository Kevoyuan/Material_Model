def gen_batch_post(foldername, sub_folders):
    # generate a batch file to extract data at one time
    with open(f"./{foldername}/post_command.bat", "w") as f:
        f.write("@echo off\n\n")

        for i in sub_folders:
            i = foldername + "\\" + str(i)
            dir = str("Z:\MA\Material_Model" + "\\" + i)
            command_line = f'start /D "{dir}" post_command.bat'
            # print(command_line)
            f.write(f'\nstart /D "{dir}" post_command.bat\n\n')

        f.write("\n\necho *** FINISHED WITH POST COMMAND SCRIPT ***")
    print("\nBatch File generated!\n")
