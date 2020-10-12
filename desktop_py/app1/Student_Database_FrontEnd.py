#frontend
from tkinter import*
import tkinter.messagebox
import stdDatabase_BackEnd

class Student:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Management System")
        self.root.geometry("1350x750+0+0")
        self.root.config(bg="Ghost White")

        StdID = StringVar()
        Firstname = StringVar()
        Lastname = StringVar()
        DoB = StringVar()
        Age = StringVar()
        Gender = StringVar()
        Address = StringVar()
        Mobile = StringVar()

        #==============Function================
        def iExit():
            iExit = tkinter.messagebox.askyesno("Student Database Management Systems", "Confirm if you want to exit")
            if iExit > 0:
                root.destroy()
                return

        def  clearData():
            self.txtStdID.delete(0,END)
            self.txtfna.delete(0,END)
            self.txtlna.delete(0,END)
            self.txtdob.delete(0,END)
            self.txtage.delete(0,END)
            self.txtgender.delete(0,END)
            self.txtaddress.delete(0,END)
            self.txtmobile.delete(0,END)

        def addData():
            stdDatabase_BackEnd.studentData()
            if(len(StdID.get()) != 0):
                stdDatabase_BackEnd.addStdRec(StdID.get(), Firstname.get(), Lastname.get(), DoB.get(), Age.get(), Gender.get(), Address.get(), Mobile.get())
                studentlist.delete(0, END)
                studentlist.insert(END, (StdID.get(), Firstname.get(), Lastname.get(), DoB.get(), Age.get(), Gender.get(), Address.get(), Mobile.get()))


        def displayData():
            studentlist.delete(o, END)
            for row in stdDatabase_BackEnd.viewData():
                studentlist.insert(END, row, str(""))

        def StudentRec(event):
            global sd
            searchStd = studentlist.curselection()[0]
            sd = studentlist.get(searchStd)

            self.txtStdID.delete(0,END)
            self.txtStdID.insert(END, sd[1])
            self.txtfna.delete(0,END)
            self.txtfna.insert(END, sd[2])
            self.txtlna.delete(0,END)
            self.txtlna.insert(END, sd[3])
            self.txtdob.delete(0,END)
            self.txtdob.insert(END, sd[4])
            self.txtage.delete(0,END)
            self.txtage.insert(END, sd[5])
            self.txtgender.delete(0,END)
            self.txtgender.insert(END, sd[6])
            self.txtaddress.delete(0,END)
            self.txtaddress.insert(END, sd[7])
            self.txtmobile.delete(0,END)
            self.txtmobile.insert(END, sd[8])

        def deleteData():
            if(len(StdID.get()) != 0):
                stdDatabase_BackEnd.deleteRec(sd[0])
                clearData()
                displayData()


        def searchDatabase():
            studentlist.delete(0, END)
            for row in stdDatabase_BackEnd.searchData(StdID.get(), Firstname.get(), Lastname.get(), DoB.get(), Age.get(), Gender.get(), Address.get(), Mobile.get()):
                studentlist.insert(END,row,str(""))


        def updateData():
            if(len(StdID.get()) != 0):
                stdDatabase_BackEnd.deleteRec(sd[0])
            if(len(StdID.get()) != 0):
                stdDatabase_BackEnd.addStdRec(StdID.get(), Firstname.get(), Lastname.get(), DoB.get(), Age.get(), Gender.get(), Address.get(), Mobile.get())
                studentlist.delete(0, END)
                studentlist.insert(END,(StdID.get(), Firstname.get(), Lastname.get(), DoB.get(), Age.get(), Gender.get(), Address.get(), Mobile.get()))

        #==============Frames================
        MainFrame = Frame(self.root, bg="Ghost White")
        MainFrame.grid()

        TitFrame = Frame(MainFrame, bd=2,padx=54, pady=8,bg="Ghost White")
        TitFrame.pack(side=TOP)

        self.lblTit = Label(TitFrame, font=('Times Roman', 47, 'bold'), text="Student Management System", bg='Ghost White')
        self.lblTit.grid()


        ButtonFrame = Frame(MainFrame, bd=1, width=1350, height=70, padx=20, pady=20,bg="Ghost White", relief=RIDGE)
        ButtonFrame.pack(side=BOTTOM)

        DataFrame = Frame(MainFrame, bd=1, width=1350, height=40, padx=20,pady=20,bg="Ghost White")
        DataFrame.pack(side=BOTTOM)

        DataFrameLEFT = LabelFrame(DataFrame, bd=1, width=850, height=600, padx=20,pady=30, bg="Ghost White", font=('arial', 12), text="Student Information")
        DataFrameLEFT.pack(side=LEFT)

        DataFrameCENTER = LabelFrame(DataFrame, bd=0, width=50, bg="Ghost White")
        DataFrameCENTER.pack(side=LEFT)

        DataFrameRIGHT = LabelFrame(DataFrame, bd=1, width=400, height=600, padx=20,pady=5,bg="Ghost White", font=('arial', 12), text="Student Details")
        DataFrameRIGHT.pack(side=RIGHT)

        #==============Labels and Entry Widgets================

        self.lblStdID = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Student ID:", padx=2, pady=2, bg='Ghost White')
        self.lblStdID.grid(row=0,column=0, sticky=W)

        self.txtStdID = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=StdID, width=39)
        self.txtStdID.grid(row=0,column=1,pady=3)



        self.lblfna = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Firstname:", padx=2, pady=2, bg='Ghost White')
        self.lblfna.grid(row=1,column=0, sticky=W)

        self.txtfna = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=Firstname, width=39)
        self.txtfna.grid(row=1,column=1,pady=3)



        self.lbllna = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Lastname:", padx=2, pady=2, bg='Ghost White')
        self.lbllna.grid(row=2,column=0, sticky=W)

        self.txtlna = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=Lastname, width=39)
        self.txtlna.grid(row=2,column=1,pady=3)



        self.lbldob = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Date of birth:", padx=2, pady=2, bg='Ghost White')
        self.lbldob.grid(row=3,column=0, sticky=W)

        self.txtdob = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=DoB, width=39)
        self.txtdob.grid(row=3,column=1,pady=3)


        self.lblage = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Age:", padx=2, pady=2, bg='Ghost White')
        self.lblage.grid(row=4,column=0, sticky=W)

        self.txtage = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=Age, width=39)
        self.txtage.grid(row=4,column=1,pady=3)



        self.lblgender = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Gender:", padx=2, pady=2, bg='Ghost White')
        self.lblgender.grid(row=5,column=0, sticky=W)

        self.txtgender = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=Gender, width=39)
        self.txtgender.grid(row=5,column=1,pady=3)



        self.lbladdress = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Address:", padx=2, pady=2, bg='Ghost White')
        self.lbladdress.grid(row=6,column=0, sticky=W)

        self.txtaddress = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=Address, width=39)
        self.txtaddress.grid(row=6,column=1,pady=3)


        self.lblmobile = Label(DataFrameLEFT, font=('arial', 15, 'bold'), text="Mobile:", padx=2, pady=2, bg='Ghost White')
        self.lblmobile.grid(row=7,column=0, sticky=W)

        self.txtmobile = Entry(DataFrameLEFT, font=('arial', 20, 'bold'), textvariable=Mobile, width=39)
        self.txtmobile.grid(row=7,column=1,pady=3)


        #==============List box and scrollbar================
        scrollbar = Scrollbar(DataFrameRIGHT)
        scrollbar.grid(row=0,column=1,sticky='ns')

        studentlist = Listbox(DataFrameRIGHT,width=41,height=15, font=('arial', 12, 'bold'), yscrollcommand=scrollbar.set)
        studentlist.bind('<<ListboxSelect>>', StudentRec)
        studentlist.grid(row=0,column=0,padx=8,pady=20)
        scrollbar.config(command=studentlist.yview)

        #==============Button widget=====================
        self.btnAddData = Button(ButtonFrame, text="Add New", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=addData)
        self.btnAddData.grid(row=0,column=0)

        self.btnDisplayData = Button(ButtonFrame, text="Display", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=displayData)
        self.btnDisplayData.grid(row=0,column=1)

        self.btnClearData = Button(ButtonFrame, text="Clear", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=clearData)
        self.btnClearData.grid(row=0,column=2)

        self.btnDeleteData = Button(ButtonFrame, text="Delete", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=deleteData)
        self.btnDeleteData.grid(row=0,column=3)

        self.btnSearchData = Button(ButtonFrame, text="Search", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=searchDatabase)
        self.btnSearchData.grid(row=0,column=4)

        self.btnUpdateData = Button(ButtonFrame, text="Update", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=updateData)
        self.btnUpdateData.grid(row=0,column=5)

        self.btnExitData = Button(ButtonFrame, text="Exit", font=('arial', 20, 'bold'),width=10,height=1,bd=4, command=iExit)
        self.btnExitData.grid(row=0,column=6)




if __name__=='__main__':
    root = Tk()
    application = Student(root)
    root.mainloop()
