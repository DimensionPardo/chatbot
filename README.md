# Chatbot IA
Este chatbot funciona con Inteligencia Artificial.
Recibe una serie de parámetros del archivo intents.json del que recibe los distintos tipos a los que puede pertenecer el mensaje. En el archvio training se reestructuran estos datos en función de unos y ceros y finalmente en el archivo chatbot se hace el mismo proceso con un mensaje introducido por el usuario y se hace la predicción en base al modelo

Para probarlo hacer lo siguiente:
1. Modificar los patrones y las respuestas en el archivo intents.json
2. Entrenar el modelo en el archivo training.py, modificando los parámetros adecuados en el modelo y el optimizador
3. Ejecutar el archivo chatbot.py y a disfrutar!
