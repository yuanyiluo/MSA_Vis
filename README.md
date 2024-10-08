## msa_vis read

This is an application that aims to provide users with a real-time
human-computer interacting environment based on sentiment understanding.

The overall framework is shown as follow:
<img alt="framework" src="D:\Research\code\MSA_Vis\imgs\figure1.png"/>

### run the app

``` python
    cd MSA_Vis/Control
    #create virtul enviroment or activate existed one 
    conda create --prefix ./venv python=3.12
    conda activate ./venv
    pip install -r requirement.txt
    python controller.py
```

### connect with llm

- **open the ollama:** open the local ollama app

``` python
    ollama run llama3.1
```

- **connect the llama with your own application**

### Some Technology

- **Insert a new thread**(for UI update timely)

````
    # controller.py (show the user widget firstly and then show the machine response)
    thread = ModelThread(self.text, self.sentiment) # function(QThread)
    thread.response_signal.connect(self.create_machine_widget)  # 处理线程返回的结果
    thread.start()
````

### Transform UI into python file

```python
    pyuic5
vis.ui - o
vis.py
```