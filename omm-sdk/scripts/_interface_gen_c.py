import sys
import json
import os
import textwrap

def GetWhitespace(indent):
    s = ""
    for i in range(0, indent):
        s = s + "   "
    return s

def WriteComment(inComment, ind):
    ws = GetWhitespace(ind)
    comment = ""
    for manualLine in inComment.splitlines():
        autoLine = textwrap.fill(manualLine, 120)
        for line in autoLine.splitlines():
            print(ws + "// " + line)

def GetNameWithPrefix(prefixLut, key, name):
    if prefixLut.get(key):
        return prefixLut[key] + name
    return name

def WriteEnum(dic, prefixLut, key, obj, ind):
    values = obj["values"]

    ws = GetWhitespace(ind)
    ws1 = GetWhitespace(ind + 1)
    name = GetNameWithPrefix(prefixLut, key, obj["name"])

    maxLen = 0
    for val in values:
        tp = name + "_" + val["name"]
        maxLen = max(len(tp), maxLen)

    declLine = "typedef enum " + name
    print(ws + declLine)
    print(ws + "{")
    first = True
    for val in values:
        if not first and obj.get("injectNewLine") and obj["injectNewLine"]:
            print("")
        tp = name + "_" + val["name"]
        if val.get("comment"):
            WriteComment(val["comment"], ind + 1)

        if val.get("value"):
            val = GetNameWithPrefix(prefixLut, val["value"], val["value"])
            args = (tp, "=", val, ",")
            print(ws1 + ('{0:<' + str(maxLen + 1) + '}{1} {2}{3}').format(*args))
        else:
            print(ws1 + tp + ",")
        first = False
    print(ws + "} " + name + ";")
    if obj.get("isFlag") and obj["isFlag"]:
        print(ws + "OMM_DEFINE_ENUM_FLAG_OPERATORS(" + name + ");")

# returns everything before the member name
def GetStructMemberLHS(dic, prefixLut, val):
    typeKey = val["type"]
    typeName = typeKey
    if dic.get(typeKey):
        typeName = dic[typeKey]["name"]

    tp = GetNameWithPrefix(prefixLut, typeKey, typeName)

    if tp == "bool":
        tp = "ommBool"

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

    return prefix + tp + postFix

def WriteStructMembers(dic, prefixLut, members, ind):
    ws = GetWhitespace(ind)

    maxLen = 0
    for val in members:
        lhs = GetStructMemberLHS(dic, prefixLut,val)
        maxLen = max(maxLen, len(lhs))

    for val in members:
        typeKey = val["type"]
        typeName = typeKey
        if dic.get(typeKey):
            typeName = dic[typeKey]["name"]

        tp = GetNameWithPrefix(prefixLut, typeKey, typeName)

        if tp == "bool":
            tp = "ommBool"

        if val.get("comment"):
            WriteComment(val["comment"], ind)

        lhs = GetStructMemberLHS(dic, prefixLut, val)

        arg = (ws, lhs, val["name"], ";")
        lineStr = ('{0}{1:<' + str(maxLen + 1) + '}{2}{3}').format(*arg)
        print(lineStr)

def WriteStructMembersDefaultValue(dic, prefixLut, members, ind):
    ws = GetWhitespace(ind)

    maxLen = 0
    for val in members:
        maxLen = max(len(val["name"]), maxLen)

    for val in members:
        if val.get("value"):
            value = val["value"]

            if value == "nullptr":
                value = "NULL"     
            if value == "true":
                value = "1"    
            if value == "false":
                value = "0"

            if isinstance(value, str):
                if value == "default":
                    typeKey = val["type"]
                    typeName = typeKey
                    tp = GetNameWithPrefix(prefixLut, typeKey, typeName)
                    value = tp + "Default()"

                args = ("v." + val["name"], " = ",  value + ";")
                lineStr = ('{0:<' + str(maxLen + 3) + '}{1}{2}').format(*args)
                print(ws + lineStr)
            elif value.get("type"):
                typeKey = val["type"]
                typeName = typeKey
                if dic.get(typeKey):
                    typeName = dic[typeKey]["name"]

                tp = GetNameWithPrefix(prefixLut, typeKey, typeName)

                args = ("v." + val["name"], " = ",  tp + "_" + value["value"] + ";")
                lineStr = ('{0:<' + str(maxLen + 3) + '}{1}{2}').format(*args)
                print(ws + lineStr)

def AnyMemberHasDefaultValue(members):
    for val in members:
        if val.get("value"):
            return True
    return False

def WriteStructMembersDefaultInit(dic, prefixLut, obj, name, ind):

    members = obj.get("members")
    if not AnyMemberHasDefaultValue(members):
        return

    ws = GetWhitespace(ind)
    ws1 = GetWhitespace(ind + 1)
    if members:
        print(ws + "\ninline " + name + " "  + name + "Default()")
        print(ws + "{")
        print(ws1 + name + " v;")
        WriteStructMembersDefaultValue(dic, prefixLut,  obj["members"], ind + 1)
        print(ws1 + "return v;")
        print(ws + "}")

def WriteStruct(dic, prefixLut, obj, ind):
    ws = GetWhitespace(ind)
    ws1 = GetWhitespace(ind + 1)
    if obj.get("cppOnly"):
        return

    if obj.get("comment"):
        WriteComment(obj["comment"], ind)

    name = GetNameWithPrefix(prefixLut, obj["name"], obj["name"])

    print(ws + "typedef struct " + name)
    print(ws + "{")
    if obj.get("members"):
       WriteStructMembers(dic, prefixLut, obj["members"], ind + 1);
    if obj.get("union_members"):
        print("")
        print(ws1 + "union")
        print(ws1 + "{")
        WriteStructMembers(dic, prefixLut,  obj["union_members"], ind + 2);
        print(ws1 + "};")
    print(ws + "} " + name + ";")

    WriteStructMembersDefaultInit(dic, prefixLut, obj, name, ind);

def WriteFunction(dic, prefixLut, key, obj, ind):
    ws = GetWhitespace(ind)
    if obj.get("comment"):
        WriteComment(obj["comment"], ind)

    prefix = ""
    ret = obj["ret"]
    lineStr = "OMM_API "
    if ret.get("const") and ret["const"]:
        lineStr += "const "
    lineStr += GetNameWithPrefix(prefixLut, ret["type"], ret["type"])
    if ret.get("ref") and ret["ref"]:
        lineStr += "*"
    lineStr += " " + prefix +  GetNameWithPrefix(prefixLut, key, obj["name"]);
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

            tp = arg["type"]
            prefix = ""
            if prefixLut.get(tp):
                prefix = prefixLut[tp]

            argStr += prefix + tp
            if arg.get("ref") and arg["ref"]:
                argStr += "*"
            if arg.get("ptr") and arg["ptr"]:
                argStr += "*"
            if arg.get("ptr2x") and arg["ptr2x"]:
                argStr += "**"

            argStr += " " + arg["name"]

    lineStr += argStr + ");"
    print(ws + lineStr)    


def WriteNamespace(dic, prefixLut, obj, ind):
    name = obj["name"]
    ws = GetWhitespace(ind)
    values = obj["values"]
    first = True
    for val in values:
        if not first:
            print("")
        first = False
        WriteObject(dic, prefixLut,val, ind)

def WriteTypedef(dic, prefixLut,obj, ind):
    name = obj["name"]
    name = GetNameWithPrefix(prefixLut, name, name)
    underlying_type = obj["underlying_type"]
    underlying_type = GetNameWithPrefix(prefixLut, underlying_type, underlying_type)
    ws = GetWhitespace(ind)
    prefix = ""
    print(ws + "typedef " + prefix + underlying_type + " " + name + ";")

def WriteObject(dic, prefixLut, key, ind):
    obj = dic[key]
    objType = obj["type"]
    if objType == "namespace":
        WriteNamespace(dic, prefixLut, obj, ind)
    if objType == "struct":
        WriteStruct(dic, prefixLut, obj, ind)
    if objType == "enum":
        WriteEnum(dic, prefixLut, key, obj, ind)     
    if objType == "function":
        WriteFunction(dic, prefixLut, key, obj, ind)     
    if objType == "typedef":
        WriteTypedef(dic, prefixLut,obj, ind)

def BuildTypeToPrefix(prefix, layout, prefixLut):
    for key in layout:
        obj = dic[key]
        objType = obj["type"]
        objName = obj["name"]
        if objType == "namespace":
            values = obj["values"]
            BuildTypeToPrefix(prefix + objName, values, prefixLut)
        if objType == "struct":
            prefixLut[key] = prefix
        if objType == "enum":
            prefixLut[key] = prefix
        if objType == "function":
            prefixLut[key] = prefix
        if objType == "typedef":
            prefixLut[key] = prefix

# first write c++ header

dirStr = sys.argv[1]

hFile = os.path.join(dirStr, "include", "omm.h")

with open(hFile, 'w') as fo:

    header_file = open( os.path.join(dirStr, "scripts", "omm_header_c.txt"), "r")
    json_file = open( os.path.join(dirStr, "scripts", "omm.json"), "r")
    dic = json.load(json_file)
    layout = dic["layout"]

    prefixLut = {}
    BuildTypeToPrefix("", layout, prefixLut)

    sys.stdout = fo # Change the standard output to the file we created.

    headStr = header_file.read()
    print(headStr)
    print("\ntypedef uint8_t ommBool;\n")

    for key in layout:

        WriteObject(dic, prefixLut, key, 0)
    print("#endif // #ifndef INCLUDE_OMM_SDK_C")
    json_file.close()
    fo.close()
