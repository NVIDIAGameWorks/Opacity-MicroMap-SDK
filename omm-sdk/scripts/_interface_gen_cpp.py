import sys
import os
import json
import textwrap

def GetWhitespace(indent):
    s = ""
    for i in range(0, indent):
        s = s + "   "
    return s

def WriteComment(inComment, ind):
    ws = GetWhitespace(ind)
    comment = textwrap.fill(inComment, 120)
    for line in comment.splitlines():
        print(ws + "// " + line)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

def GetNameWithMinimalNamespace(prefixLut, ref, nameKey, name):
    if prefixLut.get(nameKey) and prefixLut.get(ref):
        a = prefixLut[ref]
        b = prefixLut[nameKey]
        ns = intersection(b, a)
        if len(ns) != 0:
            res = ""
            isFirst = True
            for val in ns:
                if not isFirst:
                    res += "::"
                res += str(val)
                isFirst = False
            return res + "::" + name
    return name

def WriteEnum(dic, obj, ind):
    values = obj["values"]

    ws = GetWhitespace(ind)
    ws1 = GetWhitespace(ind + 1)
    name = obj["name"]
    declLine = "enum class " + name

    maxLen = 0
    for val in values:
        maxLen = max(len(val["name"]), maxLen)

    print(ws + declLine)
    print(ws + "{")
    first = True
    for val in values:
        if not first and obj.get("injectNewLine") and obj["injectNewLine"]:
            print("")
        tp = val["name"]
        if val.get("comment"):
            WriteComment(val["comment"], ind + 1)

        if val.get("value"):
            args = (tp, "=", val["value"], ",")
            print(ws1 + ('{0:<' + str(maxLen + 1) + '}{1} {2}{3}').format(*args))
        else:
            print(ws1 + tp + ",")
        first = False
    print(ws + "};")
    if obj.get("isFlag") and obj["isFlag"]:
        print(ws + "OMM_DEFINE_ENUM_FLAG_OPERATORS(" + name + ");")

def GetStructMemberLHS(dic, structKey, val):
    tp = val["type"]
    name = val["name"]

    typeKey = val["type"]
    typeName = typeKey
    if dic.get(typeKey):
        typeName = dic[typeKey]["name"]

    tp = GetNameWithMinimalNamespace(prefixLut, structKey, typeKey, typeName)

    lineStr = ""
    if val.get("static"):
        lineStr += "static "
    if val.get("constexpr"):
        lineStr += "constexpr "

    prefix = ""
    if val.get("const") and val["const"]:
        prefix = "const "
    postFix = ""
    if val.get("ptr") and val["ptr"]:
        postFix = "*"
    return lineStr + prefix + tp + postFix

def WriteStructMembers(dic, structKey, members, ind):
    ws = GetWhitespace(ind)

    maxLen = 0
    maxLenName = 0
    for val in members:
        lhs = GetStructMemberLHS(dic, structKey, val)
        maxLen = max(len(lhs), maxLen)
        maxLenName = max(len(val["name"]), maxLenName)

    for val in members:

        if val.get("comment"):
            WriteComment(val["comment"], ind)

        name = val["name"]
        lhs = GetStructMemberLHS(dic, structKey, val)

        lineStr = ""
        if val.get("value"):
            value = val["value"]
            if not isinstance(value, str):
                value = value["type"] + "::" + value["value"]
            else:
                if value == "default":
                    value = "{}"

            args = (lhs, name, "=", value, ";")
            lineStr += ('{0:<' + str(maxLen + 1) + '}{1:<' + str(maxLenName + 1) + '} {2} {3}{4}').format(*args)
        else:
            args = (lhs, name, ";")
            lineStr += ('{0:<' + str(maxLen + 1) + '}{1}{2}').format(*args)
        print(ws + lineStr)

def WriteStruct(dic, key, obj, ind):
    ws = GetWhitespace(ind)
    ws1 = GetWhitespace(ind + 1)
    if obj.get("comment"):
        WriteComment(obj["comment"], ind)

    print(ws + "struct " + obj["name"])
    print(ws + "{")
    if obj.get("members"):
       WriteStructMembers(dic, key, obj["members"], ind + 1);
    if obj.get("union_members"):
        print("")
        print(ws1 + "union")
        print(ws1 + "{")
        WriteStructMembers(dic, key, obj["union_members"], ind + 2);
        print(ws1 + "};")

    print(ws + "};")

def WriteFunction(dic, prefixLut, key, obj, ind):
    ws = GetWhitespace(ind)
    if obj.get("comment"):
        WriteComment(obj["comment"], ind)

    ret = obj["ret"]
    lineStr = "static inline "
    if ret.get("const") and ret["const"]:
        lineStr += "const "
    lineStr += ret["type"]
    if ret.get("ref") and ret["ref"]:
        lineStr += "&"
    lineStr += " " + obj["name"];
    lineStr += "("

    argStr = ""
    isFirst = True
    if obj.get("args"):
        for arg in obj["args"]:
            if not isFirst:
                argStr += ", "
            isFirst = False
            if arg.get("const") and arg["const"]:
                argStr += "const "

            tn = arg["type"];
            argStr += GetNameWithMinimalNamespace(prefixLut, key, tn, tn)
            if arg.get("ref") and arg["ref"]:
                argStr += "&"
            if arg.get("ptr") and arg["ptr"]:
                argStr += "*"
            if arg.get("ptr2x") and arg["ptr2x"]:
                argStr += "**"
            argStr += " " + arg["name"]

    lineStr += argStr + ");"
    print(ws + lineStr)    


def WriteNamespace(dic, prefixLut,obj, ind):
    name = obj["name"]
    ws = GetWhitespace(ind)
    print(ws + "namespace " + name)
    print(ws + "{\n")
    values = obj["values"]
    first = True
    for val in values:
        if not first:
            print("")
        first = False
        WriteObject(dic,prefixLut, val, ind + 1)
    print("\n" + ws + "} // namespace " + name)

def WriteTypedef(dic, obj, ind):
    name = obj["name"]
    underlying_type = obj["underlying_type"]
    ws = GetWhitespace(ind)
    print(ws + "using " + name + " = " + underlying_type + ";")

def WriteObject(dic,  prefixLut, key, ind):
    obj = dic[key]
    objType = obj["type"]
    if objType == "namespace":
        WriteNamespace(dic, prefixLut,obj, ind)
    if objType == "struct":
        WriteStruct(dic, key, obj, ind)
    if objType == "enum":
        WriteEnum(dic, obj, ind)     
    if objType == "function":
        WriteFunction(dic, prefixLut, key, obj, ind)     
    if objType == "typedef":
        WriteTypedef(dic, obj, ind)

def BuildTypeToPrefix(prefix, layout, prefixLut):
    for key in layout:
        obj = dic[key]
        objType = obj["type"]
        objName = obj["name"]
        if objType == "namespace":
            values = obj["values"]
            prefixcpy = prefix.copy()
            prefixcpy.append(objName)
            BuildTypeToPrefix(prefixcpy, values, prefixLut)
        if objType == "struct":
            prefixLut[key] = prefix
        if objType == "enum":
            prefixLut[key] = prefix
        if objType == "function":
            prefixLut[key] = prefix
        if objType == "typedef":
            prefixLut[key] = prefix

dirStr = sys.argv[1]

hppFile = os.path.join(dirStr, "include", "omm.hpp")

with open(hppFile, 'w') as fo:
    sys.stdout = fo # Change the standard output to the file we created.
   
    header_file = open( os.path.join(dirStr, "scripts", "omm_header_cpp.txt"), "r")
    footer_file = open( os.path.join(dirStr, "scripts", "omm_footer_cpp.txt"), "r")
    json_file = open( os.path.join(dirStr, "scripts", "omm.json"), "r")
    dic = json.load(json_file)
    layout = dic["layout"]

    prefixLut = {}
    BuildTypeToPrefix([], layout, prefixLut)

    headStr = header_file.read()
    print(headStr)

    for key in layout:
        WriteObject(dic, prefixLut,  key, 0)

    footerStr = footer_file.read()
    print(footerStr)
    json_file.close()
    fo.close()
