**Importante:** Durante el desarrollo para agilizar el debug se ejecutó el tratamiento de datos con el argumento **fast_debug = True** en la función transform data,

que lo que hace es seleccionar 5 clientes del total y continuar el tratamiento de datos solo con ellos. Cuando dicha opción está en **False**  se utilizan todos los datos, pero  se vio de que el docker puede

tener problemas de ram al ejecutarse según la memoria que tenga asignada docker desktop, por lo que si obtienen algún error puede ser que se deba a ello y se debe aumentar

la memoria dedicada a la aplicación. Para una evaluación más fácil y rápida se dejo **fast_debug** activado por defecto, pero se puede desactivar.
