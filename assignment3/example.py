import json
line_count = 0



for line in open ('user.json', 'r'):
    line_count+=1
    if line_count >= 10: break
    print (line)
print('ssssssss')
exit(0)


with open('user.json') as json_file:
    data = json_file.read()

d = json.loads(json_file[0])
print(d)