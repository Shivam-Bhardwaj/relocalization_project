test = open("test.txt", "r").readlines()
test_ = open("test_format.txt", "w")
for word in test:
    print(word)
    test_.write(word.rstrip())
    test_.write("\n")
