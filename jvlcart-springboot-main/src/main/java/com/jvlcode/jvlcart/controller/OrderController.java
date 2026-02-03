package com.jvlcode.jvlcart.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.jvlcode.jvlcart.dto.CreateOrderRequest;
import com.jvlcode.jvlcart.dto.OrderCreated;
import com.jvlcode.jvlcart.entity.Order;
import com.jvlcode.jvlcart.service.OrderService;

@RestController
@RequestMapping("/api/orders")
public class OrderController {
	@Autowired
	private OrderService orderService;
	
	@PostMapping
	public ResponseEntity<?> createOrder(@RequestBody CreateOrderRequest orderRequest) {
		OrderCreated orderCreated = orderService.createOrder(orderRequest);
		return ResponseEntity.ok().body(orderCreated);
	}
	
	@GetMapping("/{referenceId}")
	public  ResponseEntity<?> getOrder(@PathVariable String referenceId) {
		Order order = orderService.getOrder(referenceId);
		return ResponseEntity.ok().body(order);
	}
}
