# Here are notes about all the codes one by one

## First you can check my program:

<details>
  <summary style="font-weight:bold;">description</summary>
  <p>
My program in **++New.py** automates the creation of new Python files using a numbering scheme. It does the following:
<br>
<ul Type="square">
<li>Scans a specified directory for files named in the format `luXXX.py` (where `XXX` are three digits).
</li><li>Determines the file with the highest number.
</li><li>If no such files exist, it creates `lu001.py` as the first file.
</li><li>If a file is found, it creates a new file by incrementing the highest number.
</li><li>Paste the last copied thing in the clipboard to the new file.
</li><li>Prints a confirmation message about the new file creation.
</li></ul>
<br>This approach streamlines versioning and file management for your project files.
 </p>
</details>

<br><br><br>

---

## --- First Approach ---

---

### lu001.py

- <span style="color: red;">Some important details are not converting</span>
- <span style="color: red;">If we drew an emoji, only the circle around the face is drown</span>

---

---

### lu002.py

- <span style="color: red;">Repeats the same pattern</span>
- <span style="color: red;">Drew things away from the center</span>

---

---

### lu003.py

####  Fixes:

- <span style="color: green;">Ignores the objects away from the center</span>

####  Problems:

- <span style="color: red;">Repeats the same pattern</span>
- <span style="color: red;">Misses some details</span>

---

---

### lu004.py

####  Fixes:

- <span style="color: green;">Ignores the objects away from the center</span>
- <span style="color: green;">Doesn't repeat the same pattern many times</span>

####  Problems:

- <span style="color: red;">Misses the details that are part of the object but not in the center</span>

---

---

### lu005.py

####  Fixes:

- <span style="color: green;">Ignores the objects away from the center</span>
- <span style="color: green;">Draws the details that are part of the object but not in the center</span>

####  Problems:

- <span style="color: red;">Repeats the same pattern</span>

---

---

### lu006.py

- Same as lu005.py
- Continuous Process
- NO file creation

---

---

### lu007.py

- Same as lu006.py
- Repetition problem <span style="color:green;">fixed</span>

---

---

### lu008.py

- Control Window is added to get best results

---

---

### lu009.py

- Using Phone's Camera

---

---

### lu010.py

- Drawing on Line-US

<br><br><br>

---

## --- Second Approach ---

---

### lu011.py & lu012.py

- Using YOLOv8 to detect objects
- Using contours to draw objects
- <span style="color:red;"> Poor conversion to canvas </span>
- <span style="color:red;"> Only few objects are detected </span>

---

---

### lu013.py & lu014.py

- <span style="color:green;"> Improved </span> detection of objects
- Used Phone's Camera

---

-------------------------

---

### lu015.py

####  Features
- Using Object Detection - YOLOv3-tiny
- Using contours to draw objects
- Drawing on Line-US
- Using Phone's Camera

####  Short-comes
- <span style="color: red;">Too few shapes</span>

---

---

### lu016.py

- Using Object Detection - YOLOv3
- <span style="color:#DAA520;">More objects can be detected</span>
- <span style="color:red;">So slow</span>

<br><br><br>

---

## --- Third Approach ---

---

### lu017.py

- Database based approach
- Drawing the selected shape on Line-US

---

---

### lu018.py

- Detects the objects with YOLOv3-tiny
- Selects the closest object to the center
- Draws the selected shape on Line-US

- <span style="color:red;">Doesn't include scaling</span>

---

---

### lu019.py

- <span style="color:green;">Includes scaling</span>
- <span style="color:red;">Problems with scaling</span>

---

---

### lu020.py

- <span style="color:green;">Improved scaling</span>
- <span style="red;">Slow</span>
- <span style="red;">Problem with orientation</span>

---

---

### lu021.py

- <span style="color:green;">Improved speed by frame skipping</span>
- <span style="color:green;">Improved orientation</span>
- <span style="color:green;">code cleaned from unnecessary things</span>

- <span style="color:red;">still slow</span>

---

---

### lu022.py

- <span style="color:green;">Improved speed</span>
- <span style="color:green;">Improved orientation</span>
- <span style="color:green;">code cleaned from unnecessary things</span>
- <span style="color:green;">Controls removed, also extra tools removed</span>

- Draws on trigger

---

---

### lu023.py

- Draw on detection
- When busy, keep the camera running without sending to Line-US\
- When Idle, send to Line-US
- Threading

---

---

<div>
<br><br><br>
</div>

---

**All Programmed and assembled by Shaker Br.**
**Contact Me:** <span style="color:blue;">shbhky@gmail.com</span>