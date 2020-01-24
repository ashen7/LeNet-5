#encoding:utf-8
#python scons script file

current_dir = '/home/yipeng/workspace/lenet_5/'         #项目目录
third_party_library_dir = '/home/yipeng/third_party_library/' #第三方库目录

current_src = current_dir + 'src/'
current_tools = current_dir + 'tools/'

current_inc = current_dir + 'include/'
utility_inc = current_dir + './'
glog_inc = third_party_library_dir + 'glog/include'
gflags_inc = third_party_library_dir + 'gflags/include'
ffmpeg_inc = third_party_library_dir + 'ffmpeg/include'
protobuf_inc = third_party_library_dir + 'protobuf/include'
opencv_inc = third_party_library_dir + 'opencv/include'

glog_lib = third_party_library_dir + 'glog/lib'
gflags_lib = third_party_library_dir + 'gflags/lib'
ffmpeg_lib = third_party_library_dir + 'ffmpeg/lib'
protobuf_lib = third_party_library_dir + 'protobuf/lib'
opencv_lib = third_party_library_dir + 'opencv/lib'

#cpp头文件路径
include_dirs = [
    current_inc, 
    utility_inc, 
    glog_inc, 
    gflags_inc, 
    ffmpeg_inc, 
    protobuf_inc, 
    opencv_inc, 
]

#cpp库文件路径
lib_dirs = [
    glog_lib, 
    gflags_lib, 
    ffmpeg_lib, 
    protobuf_lib, 
    opencv_lib, 
]

#cpp库文件  动态链接库 或者静态库
libs = [
    'glog', 
    'gflags',
    'protobuf', 
    'opencv_core', 
    'opencv_highgui', 
    'opencv_imgproc', 
    'opencv_imgcodecs', 
    'opencv_videoio', 
    'opencv_objdetect', 
    'avcodec', 
    'avformat', 
    'avutil', 
    'swscale', 
]

#链接时的标志  -Wl指定运行可执行程序时 去哪个路径找动态链接库
link_flags = [
    '-pthread', 
    '-fsanitize=address', 
    '-Wl,-rpath-link=' + ":".join(lib_dirs), 
    '-Wl,-rpath=' + ":".join(lib_dirs), 
]

#cpp的编译标志
cpp_flags = [
    '-O3',                     #更好的编译优化
    '-fsanitize=address',      #asan的选项 编译链接都要用
    '-fno-omit-frame-pointer', #堆栈跟踪  
    '-g', 
    '-std=gnu++11', 
    '-W',                      #显示所有warning
    '-Wall', 
]

lenet_source = [
    current_dir + 'train_and_evaluate_lenet5.cpp', 
    current_src + 'full_connected_layer.cpp', 
    current_src + 'neural_network.cpp', 
    current_src + 'filter.cpp', 
    current_src + 'max_pooling_layer.cpp', 
    current_src + 'convolutional_layer.cpp', 
    current_src + 'lenet.cpp', 
    current_tools + 'read_mnist.cpp'
]

train_source = [
    current_dir + 'train_lenet5.cpp', 
    current_src + 'full_connected_layer.cpp', 
    current_src + 'neural_network.cpp', 
    current_src + 'filter.cpp', 
    current_src + 'max_pooling_layer.cpp', 
    current_src + 'convolutional_layer.cpp', 
    current_src + 'lenet.cpp', 
    current_tools + 'read_mnist.cpp'
]

test_source = [
    current_dir + 'test_lenet5.cpp', 
    current_src + 'full_connected_layer.cpp', 
    current_src + 'neural_network.cpp', 
    current_src + 'filter.cpp', 
    current_src + 'max_pooling_layer.cpp', 
    current_src + 'convolutional_layer.cpp', 
    current_src + 'lenet.cpp', 
    current_tools + 'read_mnist.cpp'
]

#program生成可执行文件
video_decode = Program(
    target = 'lenet5',            #可执行文件名   -o
    source = lenet_source,        #源文件列表
    CPPPATH = include_dirs,       #头文件路径列表 -I
    LIBPATH = lib_dirs,           #库文件路径列表 -L
    LIBS = libs,                  #库文件  -l
    LINKFLAGS = link_flags,       #链接的标志  -
    CPPFLAGS = cpp_flags,         #编译的标志  -
)

#program生成可执行文件
video_decode = Program(
    target = 'train_lenet5',      #可执行文件名   -o
    source = train_source,        #源文件列表
    CPPPATH = include_dirs,       #头文件路径列表 -I
    LIBPATH = lib_dirs,           #库文件路径列表 -L
    LIBS = libs,                  #库文件  -l
    LINKFLAGS = link_flags,       #链接的标志  -
    CPPFLAGS = cpp_flags,         #编译的标志  -
)

#program生成可执行文件
video_decode = Program(
    target = 'test_lenet5',       #可执行文件名   -o
    source = test_source,         #源文件列表
    CPPPATH = include_dirs,       #头文件路径列表 -I
    LIBPATH = lib_dirs,           #库文件路径列表 -L
    LIBS = libs,                  #库文件  -l
    LINKFLAGS = link_flags,       #链接的标志  -
    CPPFLAGS = cpp_flags,         #编译的标志  -
)

#安装
#bin_path = current_dir
#Install(bin_path, "fcnn")
