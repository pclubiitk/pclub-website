import csv
# with open('list.csv') as f:
# 	secys=csv.reader(f, delimiter=',')
# 	print("secys:")
# 	for secy in secys:
# 		print(f"\t'{secy[0]}':")
# 		print(f"\t\thall: {secy[1]}")
# 		print(f"\t\tmail: {secy[2]}")
# 		print(f"\t\tnumer: {secy[3]}")
# 		print(f"\t\tinsta_id: {secy[5]}")
# 		print(f"\t\tgithub: {secy[4]}\n")

list=open('list.csv', 'r')
secys=csv.reader(list, delimiter=',')

output=open('secys.yml', 'w')

print("secys:")
output.write("secys:\n")

for secy in secys:
	print(f"\t'{secy[0]}':")
	print(f"\t\thall: {secy[1]}")
	print(f"\t\tmail: {secy[2]}")
	print(f"\t\tnumber: {secy[3]}")
	print(f"\t\tinsta_id: {secy[5]}")
	print(f"\t\tgithub: {secy[4]}\n")

	output.writelines([
		f"\t'{secy[0]}':\n",
		f"\t\thall: {secy[1]}\n",
		f"\t\tmail: {secy[2]}\n",
		f"\t\tnumber: {secy[3]}\n",
		f"\t\tinsta_id: {secy[5]}\n",
		f"\t\tgithub: {secy[4]}\n\n",
	])

output.close()
list.close()
