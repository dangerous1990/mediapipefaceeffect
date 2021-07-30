# Copyright 2020 Kanstantsin Sokal.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Converts OBJ file into `mediapipe.face_geometry.Mesh3d`.
#
# The OBJ file must be located in the same directory and named "model.obj".
# The output protobuf test will be written into "model.pbtxt".
def Main():
  file_in = open("untitled.obj", "r")
  file_out = open("model.pbtxt", "w")

  v_positions = []
  v_tex_coords = []
  faces = []

  compressed_v_ids = {}
  compressed_vs = []

  for line in file_in.readlines():
    tokens = line.strip().split(' ')
    print(tokens)
    if tokens[0] == "v":
      v_positions.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
    elif tokens[0] == "vt":
      v_tex_coords.append((float(tokens[1]), float(tokens[2])))
    elif tokens[0] == "f":
      v_position_ids = [int(token.split('/')[0]) - 1 for token in tokens[1:]]
      v_tex_coord_ids = [int(token.split('/')[1]) - 1 for token in tokens[1:]]
      v_compressed_ids = []

      for v_position_id, v_tex_coord_id in zip(v_position_ids, v_tex_coord_ids):
        key = (v_position_id, v_tex_coord_id)
        if key not in compressed_v_ids:
          compressed_v_ids[key] = len(compressed_vs)
          compressed_vs.append(key)

        v_compressed_ids.append(compressed_v_ids[key])

      for middle_id in range(1, len(v_compressed_ids) - 1):
        face = []
        face.append(v_compressed_ids[0])
        face.append(v_compressed_ids[middle_id])
        face.append(v_compressed_ids[middle_id + 1])
        faces.append(tuple(face))

  file_out.write("vertex_type: VERTEX_PT\n")
  file_out.write("primitive_type: TRIANGLE\n")

  for v_position_id, v_tex_coord_id in compressed_vs:
    v_position = v_positions[v_position_id]
    v_tex_coord = v_tex_coords[v_tex_coord_id]

    line = ""
    line += "vertex_buffer: {:.6f} ".format(v_position[0])
    line += "vertex_buffer: {:.6f} ".format(v_position[1])
    line += "vertex_buffer: {:.6f} ".format(v_position[2])
    line += "vertex_buffer: {:.6f} ".format(v_tex_coord[0])
    line += "vertex_buffer: {:.6f}\n".format(1 - v_tex_coord[1])
    file_out.write(line)

  for face in faces:
    line = ""
    line += "index_buffer: {} ".format(face[0])
    line += "index_buffer: {} ".format(face[1])
    line += "index_buffer: {}\n".format(face[2])
    file_out.write(line)

  file_in.close()
  file_out.close()

if __name__ == "__main__":
  Main()
