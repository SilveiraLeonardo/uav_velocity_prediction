import os
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = os.listdir("datasets")

print("[INFO] reading files...")
velocities = []
for dir in dataset_dir:
    files = os.listdir("datasets/{}".format(dir))
    for file in files:
        if "notes" in file:
            note_files = os.listdir("datasets/{}/{}".format(dir,file))
            for notes in note_files:
                # print("datasets/{}/{}/{}".format(dir,file,notes))
                with open("datasets/{}/{}/{}".format(dir,file,notes)) as f:
                    v = np.zeros(shape=(2,1))
                    j = 0
                    for i,line in enumerate(f):
                        if i>=26 and i<=27:
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            # print(comma_index)
                            try:
                                v[j,0] = float(vel[:(comma_index+3)])
                            except:
                                v[j,0] = float(vel[:(comma_index+1)])
                            # print(v[j,0])
                            # print(vel)
                            j = j+1
                    v_mag = np.linalg.norm(v)
                    velocities.append(v_mag)
                    if v_mag == 0:
                        print("datasets/{}/{}/{}".format(dir,file,notes))
                        print(v)

_ = plt.hist(np.asarray(velocities), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

# path = "datasets/Downtown1/notes_0/1_42_26_534336.txt"

# with open(path) as f:
#     # notes = f.readlines()[17:]
#     v = np.zeros(shape=(2,1))
#     j = 0
#     for i,line in enumerate(f):
#         if i>=26 and i<=27:
#             _, description = line.strip().split(None, 1)
#             vel = description.split()[-1]
#             comma_index = int(vel.find('.'))
#             # print(comma_index)
#             v[j,0] = float(vel[:(comma_index+3)])
#             # print(vel)
#             j = j+1
#         v_magnitude = np.linalg.norm(v)
# print("{}".format(v_magnitude))
# print(v)
