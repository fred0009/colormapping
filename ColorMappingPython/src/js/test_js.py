from string import Template
import webbrowser
import os

BASE_DIR = os.path.abspath('.')
print(BASE_DIR)
with open(BASE_DIR+"/js/3d_results.html", "rb") as f:
    tpl = Template(f.read().decode("utf-8"))


def get_result_argument_for_js():
    result = ''
    for i in range(1):
        hue, value, chroma, r, g, b = '5B',2,2,34,52,59
        result += '{'
        result += "'hue':{hue}, 'value':{value}, 'chroma':{chroma}, 'rgb':'{r},{g},{b}'".format(hue=hue
                                                    , value=value, chroma=chroma, r=r,g=g,b=b)
        result += '}, '
    result = result[:-2]
    result += ""
    return result

result = get_result_argument_for_js()
print(result)

htmlcontent = tpl.substitute(color_arg=result)
f = open('tmp.html','w')
f.write(htmlcontent)
f.close()

print(htmlcontent)

filename = 'file://'+BASE_DIR.replace(' ', '%20')+'/tmp.html'
webbrowser.get().open(filename)
# os.remove('tmp.html')

