package com.jvlcode.jvlcart.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.jvlcode.jvlcart.entity.Order;

public interface OrderRepository extends JpaRepository<Order, Long> {
   	Optional<Order> findByReferenceId(String referenceId);
}
