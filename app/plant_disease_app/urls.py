from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from django.views.generic import TemplateView
from .views import home,predict_image,about,contact,services,diseasePage,testimonial

urlpatterns = [
    path('', home, name='home'),
    path('about', about, name='about'),
    path('contact', contact, name='contact'),
    path('services', services, name='services'),
    path('diseasePage', diseasePage, name='diseasePage'),
    path('testimonial', testimonial, name='testimonial'),

    path('predict/',predict_image, name='predict_image'),  # Endpoint for image prediction


]