cmake_minimum_required(VERSION 2.8.3)
project(work_vision)

## Encontra todos os pacotes catkin necessários para compilar e executar.
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  visualization_msgs # ADICIONADO: Para os marcadores do RViz (MarkerArray)
  std_srvs           # ADICIONADO: Para os serviços (Trigger)
  cv_bridge          # ADICIONADO: Para conversão de imagens
  message_generation
)

## Configuração para scripts Python
catkin_python_setup()

###################################
## Geração de Mensagens Customizadas
###################################

## Adiciona sua nova lista de arquivos .msg da pasta 'msg/'
add_message_files(
  FILES
  Description2D.msg
  Description3D.msg
  Recognitions2D.msg
  Recognitions3D.msg
  KeyPoint2D.msg
  KeyPoint3D.msg
)

## Gera as mensagens, garantindo que as dependências delas estejam listadas.
generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   geometry_msgs
   vision_msgs
)

###################################
## Definições do Pacote
###################################

## Declara as dependências do pacote para que outros pacotes e o ambiente saibam do que ele precisa.
catkin_package(
  CATKIN_DEPENDS
    message_runtime
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    vision_msgs
    visualization_msgs # ADICIONADO
    std_srvs           # ADICIONADO
    cv_bridge          # ADICIONADO
)

## Diretórios de include (importante para C++)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

###################################
## Instalação de Arquivos
###################################

## Instala os scripts Python executáveis da pasta 'scripts/'.
## !! ESTA É A CORREÇÃO MAIS IMPORTANTE !!
install(PROGRAMS
  scripts/yolov8_node.py # <-- SUBSTITUA PELO NOME CORRETO DO SEU SCRIPT PRINCIPAL
  # Se você dividiu em detector_2d.py e estimator_3d.py, liste os dois aqui:
  # scripts/detector_2d.py
  # scripts/estimator_3d.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)