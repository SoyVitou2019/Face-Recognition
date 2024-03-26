import tkinter as tk

class BackButtonWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Window with Back Button")
        self.master.geometry("800x500")  # Set window size to 800x500

        self.frame = tk.Frame(master)
        self.frame.pack(padx=60, pady=30)

        self.back_button = tk.Button(self.frame, text="Back", command=self.go_back)
        self.back_button.pack()

    def go_back(self):
        # Add functionality to go back here
        print("Going back...")  # Placeholder action, replace with actual functionality

def button_register_click():
    back_window = BackButtonWindow(tk.Toplevel())
    back_window.master.grab_set()  # Make sure toplevel window grabs the focus

# Create the main Tkinter window
root = tk.Tk()

# Create a button to register a click
register_button = tk.Button(root, text="Register Click", command=button_register_click)
register_button.pack()

# Run the Tkinter event loop
root.mainloop()
