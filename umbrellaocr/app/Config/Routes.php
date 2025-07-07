<?php

use CodeIgniter\Router\RouteCollection;

/**
 * @var RouteCollection $routes
 */
$routes->get('/', 'Home::index');
$routes->get('/converter', 'C_Converter::index');
$routes->post('/converter/upload', 'C_Converter::upload');
$routes->post('/converter/process', 'C_Converter::process');
$routes->get('/converter/download/(:any)', 'C_Converter::download/$1');