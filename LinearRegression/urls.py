from django.urls import path
from . import views

urlpatterns = [
    # path('predict/', views.predict, name='predict'),
    path('predict/',views.make_prediction,name="makeprediction"),
    path('train/', views.train,name='train'),
    path('prediction2',views.prediction2,name="prediction2")
]